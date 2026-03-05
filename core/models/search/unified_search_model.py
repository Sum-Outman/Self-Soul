#!/usr/bin/env python3
"""
Unified Search Model - AGI-Level Search and Information Retrieval System

This module provides advanced search capabilities including:
- Keyword search with BM25 ranking
- Semantic search with vector embeddings
- Hybrid search combining multiple strategies
- Result ranking and filtering
- Query understanding and expansion
"""

import sys
import os
import logging
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
import numpy as np

# Import torch if available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# Import core modules
from core.models.unified_model_template import UnifiedModelTemplate
from core.from_scratch_training import FromScratchTrainer
from core.external_api_service import ExternalAPIService
from core.agi_tools import AGITools
from core.error_handling import error_handler
from core.unified_stream_processor import StreamProcessor

# Check for sentence-transformers availability
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# Check for rank_bm25 availability
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

# Configure logging
logger = logging.getLogger(__name__)


class InvertedIndex:
    """Simple inverted index implementation for keyword search"""
    
    def __init__(self):
        self.index = defaultdict(list)  # word -> list of (doc_id, position)
        self.documents = {}  # doc_id -> document text
        self.doc_metadata = {}  # doc_id -> metadata
        self.next_doc_id = 1
        
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> int:
        """Add a document to the index"""
        doc_id = self.next_doc_id
        self.next_doc_id += 1
        
        self.documents[doc_id] = text
        self.doc_metadata[doc_id] = metadata or {}
        
        # Tokenize and index
        tokens = self._tokenize(text)
        for position, token in enumerate(tokens):
            self.index[token].append((doc_id, position))
        
        return doc_id
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents containing query terms"""
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Calculate document scores (simple TF-based)
        doc_scores = defaultdict(float)
        for token in query_tokens:
            if token in self.index:
                # Count occurrences in each document
                doc_counts = defaultdict(int)
                for doc_id, _ in self.index[token]:
                    doc_counts[doc_id] += 1
                
                # Add to scores (log TF)
                for doc_id, count in doc_counts.items():
                    doc_scores[doc_id] += np.log1p(count)
        
        # Sort by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            results.append({
                'doc_id': doc_id,
                'text': self.documents[doc_id],
                'metadata': self.doc_metadata[doc_id],
                'score': score,
                'method': 'inverted_index'
            })
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - split on whitespace and lowercase"""
        if not text:
            return []
        
        # Convert to lowercase and split
        tokens = text.lower().split()
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def clear(self):
        """Clear the index"""
        self.index.clear()
        self.documents.clear()
        self.doc_metadata.clear()
        self.next_doc_id = 1


class BM25SearchEngine:
    """BM25 search engine implementation"""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
        
    def index_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Index a list of documents"""
        if not documents:
            return
        
        self.documents = documents
        self.doc_metadata = metadata or [{} for _ in documents]
        
        # Tokenize documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Create BM25 index
        if BM25_AVAILABLE and BM25Okapi:
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            # Fallback to simple implementation
            self.bm25 = self._create_simple_bm25(tokenized_docs)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using BM25"""
        if not self.bm25 or not self.documents:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top_k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'doc_id': idx,
                    'text': self.documents[idx],
                    'metadata': self.doc_metadata[idx],
                    'score': float(scores[idx]),
                    'method': 'bm25'
                })
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        if not text:
            return []
        
        # Simple tokenization
        tokens = text.lower().split()
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def _create_simple_bm25(self, tokenized_docs):
        """Simple BM25 implementation as fallback"""
        class SimpleBM25:
            def __init__(self, docs):
                self.docs = docs
                self.doc_lengths = [len(doc) for doc in docs]
                self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 0
                self.doc_count = len(docs)
                
                # Build term frequency dictionary
                self.term_doc_freq = defaultdict(int)
                self.term_freqs = []
                
                for doc in docs:
                    term_freq = Counter(doc)
                    self.term_freqs.append(term_freq)
                    for term in term_freq:
                        self.term_doc_freq[term] += 1
            
            def get_scores(self, query):
                scores = np.zeros(self.doc_count)
                
                if not query:
                    return scores
                
                k1 = 1.5
                b = 0.75
                
                for i in range(self.doc_count):
                    doc_score = 0.0
                    doc_length = self.doc_lengths[i]
                    
                    for term in query:
                        if term not in self.term_doc_freq:
                            continue
                        
                        # Term frequency in document
                        tf = self.term_freqs[i].get(term, 0)
                        
                        # Document frequency
                        df = self.term_doc_freq[term]
                        
                        # Inverse document frequency
                        idf = np.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
                        
                        # BM25 scoring
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * doc_length / self.avg_doc_length)
                        
                        doc_score += idf * numerator / denominator if denominator > 0 else 0
                    
                    scores[i] = doc_score
                
                return scores
        
        return SimpleBM25(tokenized_docs)


class SemanticSearchEngine:
    """Semantic search engine using sentence embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = None
        self.embeddings = None
        self.documents = []
        self.doc_metadata = []
        
        # Initialize model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded semantic search model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer model: {e}")
                self.model = None
        else:
            logger.warning("Sentence-transformers not available, semantic search disabled")
    
    def index_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Index documents with semantic embeddings"""
        if not self.model or not documents:
            return
        
        self.documents = documents
        self.doc_metadata = metadata or [{} for _ in documents]
        
        # Generate embeddings
        try:
            self.embeddings = self.model.encode(documents, show_progress_bar=False)
            logger.info(f"Indexed {len(documents)} documents with semantic embeddings")
        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}")
            self.embeddings = None
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using semantic similarity"""
        if not self.model or not self.embeddings or not self.documents:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate cosine similarities
            if self.embeddings is not None and len(self.embeddings) > 0:
                # Normalize embeddings for cosine similarity
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                
                similarities = np.dot(doc_norms, query_norm)
                
                # Get top_k results
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                results = []
                for idx in top_indices:
                    if similarities[idx] > 0:
                        results.append({
                            'doc_id': idx,
                            'text': self.documents[idx],
                            'metadata': self.doc_metadata[idx],
                            'score': float(similarities[idx]),
                            'method': 'semantic'
                        })
                
                return results
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
        
        return []
    
    def is_available(self) -> bool:
        """Check if semantic search is available"""
        return self.model is not None and self.embeddings is not None


class UnifiedSearchModel(UnifiedModelTemplate):
    """
    AGI-Level Unified Search Model
    
    Advanced search and information retrieval system with capabilities for:
    - Keyword-based search with inverted index
    - BM25 ranking for relevance scoring
    - Semantic search with vector embeddings
    - Hybrid search combining multiple strategies
    - Query understanding and expansion
    - Result ranking and filtering
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Search Model with optional pre-trained semantic search support
        
        Args:
            config: Configuration dictionary with optional keys:
                - from_scratch: bool, if True use custom search engines, if False use enhanced semantic search
                - device: str, device to load model on ('cpu' or 'cuda')
                - use_semantic_search: bool, if True use sentence-transformers (default: True)
                - semantic_model: str, sentence transformer model name (default: 'all-MiniLM-L6-v2')
                - max_documents: int, maximum number of documents to index (default: 10000)
        """
        # Call parent constructor
        if config is None:
            config = {}
        super().__init__(config)
        
        # Extract configuration
        self.from_scratch = config.get('from_scratch', False)
        self.device = config.get('device', 'cpu')
        self.use_semantic_search = config.get('use_semantic_search', True)
        self.semantic_model = config.get('semantic_model', 'all-MiniLM-L6-v2')
        self.max_documents = config.get('max_documents', 10000)
        self.is_pretrained = not self.from_scratch
        
        # Model-specific configuration
        self.model_name = "search"
        self.model_type = "search"
        self.model_id = 8028  # Port number for search model
        
        # Search capabilities
        self.search_methods = {
            'keyword': ['inverted_index', 'boolean_search', 'phrase_search'],
            'ranking': ['bm25', 'tfidf', 'custom_ranking'],
            'semantic': ['vector_search', 'embedding_similarity', 'neural_ranking'],
            'hybrid': ['combined_score', 'weighted_fusion', 'reranking']
        }
        
        # Initialize search engines
        self.inverted_index = InvertedIndex()
        self.bm25_engine = BM25SearchEngine()
        self.semantic_engine = SemanticSearchEngine(self.semantic_model)
        
        # Document storage
        self.documents = []
        self.document_metadata = []
        
        # Search history
        self.search_history = []
        self.max_history_size = 100
        
        # Initialize after super().__init__ completes
        self._initialize_search_engines(config)
        
        logger.info(f"UnifiedSearchModel initialized with ID: {self.model_id}, from_scratch: {self.from_scratch}, is_pretrained: {self.is_pretrained}")
    
    def _get_model_id(self) -> str:
        """Return model identifier"""
        return "search"
    
    def _get_model_type(self) -> str:
        """Return model type identifier"""
        return "search"
    
    def _initialize_search_engines(self, config: Dict[str, Any]):
        """Initialize search engines based on configuration"""
        try:
            logger.info(f"Initializing search engines, semantic search: {self.use_semantic_search}")
            
            # Check semantic search availability
            if self.use_semantic_search and not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("Sentence-transformers not available, disabling semantic search")
                self.use_semantic_search = False
            
            # Initialize with sample data if needed
            if config.get('initialize_with_sample', True):
                self._initialize_with_sample_data()
                
        except Exception as e:
            logger.warning(f"Search engine initialization failed: {e}")
    
    def _initialize_with_sample_data(self):
        """Initialize with sample documents for testing"""
        sample_documents = [
            "Artificial intelligence is transforming the world.",
            "Machine learning algorithms learn from data.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to see and interpret images.",
            "Reinforcement learning trains agents through trial and error.",
            "Search engines help find information on the internet.",
            "Information retrieval is the science of searching for information.",
            "Vector databases store high-dimensional embeddings.",
            "Semantic search understands the meaning behind queries."
        ]
        
        sample_metadata = [
            {"category": "AI", "source": "sample"},
            {"category": "ML", "source": "sample"},
            {"category": "DL", "source": "sample"},
            {"category": "NLP", "source": "sample"},
            {"category": "CV", "source": "sample"},
            {"category": "RL", "source": "sample"},
            {"category": "Search", "source": "sample"},
            {"category": "IR", "source": "sample"},
            {"category": "Database", "source": "sample"},
            {"category": "Search", "source": "sample"}
        ]
        
        # Index sample documents
        self.index_documents(sample_documents, sample_metadata)
        
        logger.info(f"Initialized with {len(sample_documents)} sample documents")
    
    def index_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Index documents for searching
        
        Args:
            documents: List of document texts
            metadata: List of metadata dictionaries (optional)
        """
        if not documents:
            return
        
        # Limit documents if needed
        if len(documents) > self.max_documents:
            logger.warning(f"Limiting documents from {len(documents)} to {self.max_documents}")
            documents = documents[:self.max_documents]
            if metadata:
                metadata = metadata[:self.max_documents]
        
        # Store documents
        self.documents = documents
        self.document_metadata = metadata or [{} for _ in documents]
        
        # Index in inverted index
        for i, doc in enumerate(documents):
            doc_id = i + 1  # Use 1-based indexing
            meta = self.document_metadata[i] if i < len(self.document_metadata) else {}
            self.inverted_index.add_document(doc, meta)
        
        # Index in BM25
        self.bm25_engine.index_documents(documents, metadata)
        
        # Index in semantic engine if available
        if self.use_semantic_search and self.semantic_engine.is_available():
            self.semantic_engine.index_documents(documents, metadata)
        
        logger.info(f"Indexed {len(documents)} documents")
    
    def search(self, query: str, method: str = "hybrid", top_k: int = 10, 
               filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search for documents matching the query
        
        Args:
            query: Search query string
            method: Search method ('keyword', 'bm25', 'semantic', 'hybrid')
            top_k: Number of results to return
            filters: Optional filters for results
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            start_time = time.time()
            
            # Validate query
            if not query or not query.strip():
                return {
                    'status': 'error',
                    'message': 'Empty query',
                    'results': [],
                    'metadata': {}
                }
            
            query = query.strip()
            
            # Perform search based on method
            results = []
            
            if method == "keyword":
                results = self.inverted_index.search(query, top_k)
            elif method == "bm25":
                results = self.bm25_engine.search(query, top_k)
            elif method == "semantic":
                if self.use_semantic_search:
                    results = self.semantic_engine.search(query, top_k)
                else:
                    results = self.bm25_engine.search(query, top_k)
            elif method == "hybrid":
                # Combine multiple search methods
                results = self._hybrid_search(query, top_k)
            else:
                # Default to hybrid
                results = self._hybrid_search(query, top_k)
            
            # Apply filters if provided
            if filters and results:
                results = self._apply_filters(results, filters)
            
            # Limit to top_k
            results = results[:top_k]
            
            # Add to search history
            self._add_to_search_history(query, method, len(results))
            
            # Calculate search time
            search_time = time.time() - start_time
            
            return {
                'status': 'success',
                'query': query,
                'method': method,
                'results': results,
                'count': len(results),
                'search_time': search_time,
                'metadata': {
                    'total_documents': len(self.documents),
                    'filters_applied': bool(filters),
                    'history_size': len(self.search_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'results': [],
                'metadata': {}
            }
    
    def _hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining multiple methods"""
        all_results = []
        
        # Get results from each method
        keyword_results = self.inverted_index.search(query, top_k * 2)
        bm25_results = self.bm25_engine.search(query, top_k * 2)
        
        # Combine results
        result_map = {}
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result['doc_id']
            if doc_id not in result_map:
                result_map[doc_id] = {
                    'doc_id': doc_id,
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'scores': {}
                }
            result_map[doc_id]['scores']['keyword'] = result['score']
        
        # Add BM25 results
        for result in bm25_results:
            doc_id = result['doc_id']
            if doc_id not in result_map:
                result_map[doc_id] = {
                    'doc_id': doc_id,
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'scores': {}
                }
            result_map[doc_id]['scores']['bm25'] = result['score']
        
        # Add semantic results if available
        if self.use_semantic_search:
            semantic_results = self.semantic_engine.search(query, top_k * 2)
            for result in semantic_results:
                doc_id = result['doc_id']
                if doc_id not in result_map:
                    result_map[doc_id] = {
                        'doc_id': doc_id,
                        'text': result['text'],
                        'metadata': result['metadata'],
                        'scores': {}
                    }
                result_map[doc_id]['scores']['semantic'] = result['score']
        
        # Calculate combined scores
        combined_results = []
        for doc_id, result in result_map.items():
            # Calculate weighted average score
            scores = result['scores']
            if not scores:
                continue
            
            # Simple averaging for now
            avg_score = sum(scores.values()) / len(scores)
            
            combined_results.append({
                'doc_id': doc_id,
                'text': result['text'],
                'metadata': result['metadata'],
                'score': avg_score,
                'method': 'hybrid',
                'component_scores': scores
            })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        return combined_results[:top_k]
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Check if result passes all filters
            passes = True
            
            for key, value in filters.items():
                if key in metadata:
                    if isinstance(value, list):
                        # Check if metadata value is in filter list
                        if metadata[key] not in value:
                            passes = False
                            break
                    else:
                        # Check equality
                        if metadata[key] != value:
                            passes = False
                            break
                else:
                    # Key not in metadata, fail if filter requires it
                    passes = False
                    break
            
            if passes:
                filtered_results.append(result)
        
        return filtered_results
    
    def _add_to_search_history(self, query: str, method: str, result_count: int):
        """Add search to history"""
        search_entry = {
            'timestamp': time.time(),
            'query': query,
            'method': method,
            'result_count': result_count
        }
        
        self.search_history.append(search_entry)
        
        # Limit history size
        if len(self.search_history) > self.max_history_size:
            self.search_history = self.search_history[-self.max_history_size:]
    
    def clear_index(self):
        """Clear all indexed documents"""
        self.inverted_index.clear()
        self.bm25_engine = BM25SearchEngine()
        self.semantic_engine = SemanticSearchEngine(self.semantic_model)
        
        self.documents = []
        self.document_metadata = []
        
        logger.info("Search index cleared")
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'total_documents': len(self.documents),
            'search_history_size': len(self.search_history),
            'indexed_terms': len(self.inverted_index.index),
            'semantic_search_available': self.semantic_engine.is_available() if hasattr(self, 'semantic_engine') else False,
            'bm25_available': BM25_AVAILABLE,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE
        }
    
    def forward(self, x, **kwargs):
        """Forward pass for Search Model
        
        Processes search queries through search engines.
        Supports query strings or dictionaries with search parameters.
        """
        # If input is a dictionary with query, extract it
        if isinstance(x, dict):
            query = x.get('query', '')
            method = x.get('method', 'hybrid')
            top_k = x.get('top_k', 10)
            filters = x.get('filters')
            
            if not query:
                # Try to extract query from other fields
                for key in ['text', 'input', 'search']:
                    if key in x:
                        query = str(x[key])
                        break
            
            if query:
                return self.search(query, method, top_k, filters)
            else:
                # Return empty results
                return {
                    'status': 'error',
                    'message': 'No query provided',
                    'results': []
                }
        elif isinstance(x, str):
            # Input is query string
            return self.search(x, **kwargs)
        else:
            # Try to convert to string
            try:
                query = str(x)
                return self.search(query, **kwargs)
            except:
                return {
                    'status': 'error',
                    'message': f'Unsupported input type: {type(x)}',
                    'results': []
                }
    
    def train_step(self, batch, optimizer=None, criterion=None, device=None):
        """Search model specific training step"""
        self.logger.info(f"Search model training step on device: {device if device else self.device}")
        
        # Search models typically don't have traditional training,
        # but we can simulate training for compatibility
        
        # Call parent implementation
        return super().train_step(batch, optimizer, criterion, device)
    
    def _get_supported_operations(self) -> List[str]:
        """Return list of supported operations"""
        return [
            "search",
            "index",
            "clear_index",
            "get_stats",
            "keyword_search",
            "semantic_search",
            "hybrid_search"
        ]
    
    def _process_operation(self, operation: str, data: Any, **kwargs) -> Dict[str, Any]:
        """Process specific operations for search model"""
        try:
            if operation == "search":
                query = data if isinstance(data, str) else data.get('query', '')
                return self.search(query, **kwargs)
            elif operation == "index":
                if isinstance(data, dict):
                    documents = data.get('documents', [])
                    metadata = data.get('metadata')
                    self.index_documents(documents, metadata)
                    return {
                        'status': 'success',
                        'message': f'Indexed {len(documents)} documents'
                    }
                else:
                    return {
                        'status': 'error',
                        'message': 'Index operation requires documents list'
                    }
            elif operation == "clear_index":
                self.clear_index()
                return {
                    'status': 'success',
                    'message': 'Index cleared'
                }
            elif operation == "get_stats":
                return {
                    'status': 'success',
                    'stats': self.get_search_stats()
                }
            elif operation == "keyword_search":
                query = data if isinstance(data, str) else data.get('query', '')
                return self.search(query, method='keyword', **kwargs)
            elif operation == "semantic_search":
                query = data if isinstance(data, str) else data.get('query', '')
                return self.search(query, method='semantic', **kwargs)
            elif operation == "hybrid_search":
                query = data if isinstance(data, str) else data.get('query', '')
                return self.search(query, method='hybrid', **kwargs)
            else:
                return {
                    'status': 'error',
                    'message': f'Unsupported operation: {operation}',
                    'supported_operations': self._get_supported_operations()
                }
        except Exception as e:
            logger.error(f"Operation processing failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _create_stream_processor(self):
        """Create stream processor for search model"""
        from core.unified_stream_processor import StreamProcessor
        return StreamProcessor(
            model_type="search",
            supported_operations=self._get_supported_operations(),
            config=self.config
        )
    
    def _initialize_model_specific_components(self, config: Dict[str, Any] = None) -> None:
        """Initialize search-specific components"""
        try:
            # Initialize search engines
            self._initialize_search_engines(config or {})
            
            # Set device (GPU if available)
            if TORCH_AVAILABLE:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            logger.info(f"Search model components initialized, device: {self.device}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize search components: {e}")
    
    def _initialize_search_components(self):
        """Initialize search components (alias for compatibility)"""
        self._initialize_model_specific_components()
    
    def retrieve(self, query: str, method: str = 'keyword', top_k: int = 10, **kwargs) -> Dict[str, Any]:
        """Retrieve documents for query
        
        Args:
            query: Search query string
            method: Search method ('keyword', 'semantic', 'hybrid')
            top_k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary with retrieval results
        """
        params = {
            "query": query,
            "method": method,
            "top_k": top_k,
            **kwargs
        }
        return self._process_operation("search", params)
    
    def rank(self, query: str, documents: List[str], top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Rank documents for query
        
        Args:
            query: Search query string
            documents: List of documents to rank
            top_k: Number of top documents to return
            **kwargs: Additional ranking parameters
            
        Returns:
            Dictionary with ranking results
        """
        try:
            # Simple ranking based on term frequency
            query_terms = query.lower().split()
            ranked_docs = []
            
            for i, doc in enumerate(documents):
                doc_lower = doc.lower()
                score = 0.0
                
                # Calculate term frequency score
                for term in query_terms:
                    if term in doc_lower:
                        # Simple TF scoring: count occurrences
                        count = doc_lower.count(term)
                        score += count / max(len(doc_lower.split()), 1)
                
                # Normalize score by document length
                doc_length = len(doc_lower.split())
                if doc_length > 0:
                    score = score / doc_length
                
                ranked_docs.append({
                    'document': doc,
                    'score': score,
                    'original_index': i
                })
            
            # Sort by score descending
            ranked_docs.sort(key=lambda x: x['score'], reverse=True)
            
            # Add rank position and limit to top_k
            for i, doc in enumerate(ranked_docs[:top_k]):
                doc['rank'] = i + 1
            
            return {
                'status': 'success',
                'query': query,
                'ranked_documents': ranked_docs[:top_k],
                'count': min(len(ranked_docs), top_k),
                'method': 'term_frequency',
                'total_documents': len(documents)
            }
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'ranked_documents': [],
                'count': 0,
                'total_documents': len(documents)
            }
    
    def expand_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Expand query with related terms
        
        Args:
            query: Original query string
            **kwargs: Additional expansion parameters
            
        Returns:
            Dictionary with expanded query results
        """
        try:
            # Simple query expansion: add synonyms and related terms
            # In a real implementation, this would use a thesaurus or word embeddings
            expanded_terms = [query]
            
            # Add simple synonyms for common terms
            synonyms_map = {
                'test': ['exam', 'trial', 'check'],
                'query': ['question', 'inquiry', 'request'],
                'search': ['find', 'lookup', 'seek']
            }
            
            for term, synonyms in synonyms_map.items():
                if term in query.lower():
                    expanded_terms.extend(synonyms)
            
            # Remove duplicates
            expanded_terms = list(dict.fromkeys(expanded_terms))
            
            return {
                'status': 'success',
                'original_query': query,
                'expanded_queries': expanded_terms,
                'expansion_method': 'simple_synonyms',
                'term_count': len(expanded_terms)
            }
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'expanded_queries': [query],
                'term_count': 1
            }


# Export the main class
__all__ = ['UnifiedSearchModel', 'InvertedIndex', 'BM25SearchEngine', 'SemanticSearchEngine']