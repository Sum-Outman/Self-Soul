"""
Enhanced Data Collector - Industrial-scale data collection for large model training

Provides comprehensive data collection capabilities from multiple sources:
1. Web scraping and crawling
2. Public dataset integration
3. API-based data collection
4. Synthetic data generation
5. Multi-modal data collection

Designed to support large-scale model training with TB-level datasets.
"""

import os
import json
import logging
import asyncio
import aiohttp
import requests
import time
import random
import re
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import math

try:
    import bs4
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    logging.warning("BeautifulSoup not available. Install with: pip install beautifulsoup4")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available. Install with: pip install pandas")

from core.error_handling import error_handler
from core.data_processor import DataType

logger = logging.getLogger(__name__)

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    source_type: str  # web, api, dataset, synthetic, multimodal
    url: Optional[str] = None
    api_key: Optional[str] = None
    rate_limit: int = 10  # requests per second
    max_items: int = 10000
    data_format: str = "json"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class DataCollectionStats:
    """Statistics for data collection"""
    total_collected: int = 0
    total_size_bytes: int = 0
    sources_processed: int = 0
    successful_collections: int = 0
    failed_collections: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def collection_rate(self) -> float:
        if self.duration_seconds > 0:
            return self.total_collected / self.duration_seconds
        return 0.0

class WebDataCollector:
    """Web data collector with intelligent crawling and parsing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.session = None
        self.crawled_urls = set()
        self.max_depth = self.config.get('max_depth', 3)
        self.max_pages = self.config.get('max_pages', 1000)
        self.respect_robots = self.config.get('respect_robots', True)
        self.user_agent = self.config.get('user_agent', 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
    async def initialize(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': self.user_agent},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
    async def close(self):
        """Close async session"""
        if self.session:
            await self.session.close()
            
    async def crawl_website(self, url: str, depth: int = 0) -> List[Dict[str, Any]]:
        """Crawl website and extract text content"""
        if depth > self.max_depth or len(self.crawled_urls) >= self.max_pages:
            return []
            
        if url in self.crawled_urls:
            return []
            
        self.crawled_urls.add(url)
        logger.info(f"Crawling: {url} (depth: {depth})")
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return []
                    
                html = await response.text()
                data = self.extract_content(html, url)
                
                # Extract links for further crawling
                if depth < self.max_depth:
                    links = self.extract_links(html, url)
                    tasks = []
                    for link in links[:10]:  # Limit concurrent requests
                        if link not in self.crawled_urls:
                            tasks.append(self.crawl_website(link, depth + 1))
                    
                    if tasks:
                        sub_results = await asyncio.gather(*tasks, return_exceptions=True)
                        for result in sub_results:
                            if isinstance(result, list):
                                data.extend(result)
                
                return data
                
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return []
    
    def extract_content(self, html: str, url: str) -> List[Dict[str, Any]]:
        """Extract meaningful content from HTML"""
        if not BEAUTIFULSOUP_AVAILABLE:
            return []
            
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if len(text) < 100:  # Skip very short content
                return []
                
            # Extract metadata
            title = soup.title.string if soup.title else ""
            meta_desc = soup.find("meta", attrs={"name": "description"})
            description = meta_desc["content"] if meta_desc else ""
            
            return [{
                "url": url,
                "title": title,
                "description": description,
                "content": text,
                "content_length": len(text),
                "collected_at": datetime.now().isoformat(),
                "data_type": "text",
                "source": "web_crawler"
            }]
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return []
    
    def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML"""
        if not BEAUTIFULSOUP_AVAILABLE:
            return []
            
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    from urllib.parse import urljoin
                    href = urljoin(base_url, href)
                elif not href.startswith(('http://', 'https://')):
                    continue
                    
                # Filter out non-HTML content
                if any(href.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.mp4', '.mp3']):
                    continue
                    
                links.append(href)
                
            return list(set(links))[:50]  # Limit to 50 unique links
            
        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []

class APIDataCollector:
    """API-based data collector for various data sources"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.api_clients = {}
        self.rate_limiters = {}
        
    def collect_from_openai(self, api_key: str, prompts: List[str], 
                           model: str = "gpt-3.5-turbo") -> List[Dict[str, Any]]:
        """Collect data using OpenAI API"""
        try:
            import openai
            openai.api_key = api_key
            
            results = []
            for prompt in prompts:
                try:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                        temperature=0.7
                    )
                    
                    content = response.choices[0].message.content
                    results.append({
                        "prompt": prompt,
                        "response": content,
                        "model": model,
                        "collected_at": datetime.now().isoformat(),
                        "data_type": "text",
                        "source": "openai_api"
                    })
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error generating with OpenAI: {e}")
                    continue
                    
            return results
            
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            return []
    
    def collect_from_huggingface(self, dataset_name: str, 
                                split: str = "train",
                                config_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Collect data from HuggingFace datasets"""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(dataset_name, config_name, split=split)
            results = []
            
            # Convert to list of dictionaries
            for i, item in enumerate(dataset):
                if isinstance(item, dict):
                    results.append({
                        **item,
                        "dataset_name": dataset_name,
                        "split": split,
                        "index": i,
                        "collected_at": datetime.now().isoformat(),
                        "source": "huggingface"
                    })
                elif isinstance(item, (str, int, float)):
                    results.append({
                        "text": str(item),
                        "dataset_name": dataset_name,
                        "split": split,
                        "index": i,
                        "collected_at": datetime.now().isoformat(),
                        "source": "huggingface"
                    })
                
                # Limit for demonstration
                if len(results) >= 1000:
                    break
                    
            return results
            
        except ImportError:
            logger.error("Datasets package not installed. Install with: pip install datasets")
            return []
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return []

class SyntheticDataGenerator:
    """Synthetic data generator for training large models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vocabulary = self._load_vocabulary()
        
    def _load_vocabulary(self) -> List[str]:
        """Load vocabulary for text generation"""
        # Base vocabulary
        vocab = [
            "artificial", "intelligence", "machine", "learning", "deep", "neural",
            "network", "transformer", "attention", "mechanism", "language", "model",
            "computer", "vision", "natural", "processing", "algorithm", "data",
            "science", "programming", "python", "javascript", "java", "c++", "code",
            "function", "class", "object", "method", "variable", "array", "list",
            "dictionary", "string", "integer", "float", "boolean", "null", "undefined",
            "true", "false", "if", "else", "for", "while", "return", "import", "export",
            "module", "package", "library", "framework", "application", "system",
            "database", "server", "client", "web", "mobile", "desktop", "cloud",
            "security", "privacy", "encryption", "authentication", "authorization"
        ]
        
        # Add more technical terms
        vocab.extend([
            "backpropagation", "gradient", "descent", "optimization", "loss", "function",
            "accuracy", "precision", "recall", "f1", "score", "metric", "evaluation",
            "training", "validation", "testing", "dataset", "preprocessing", "cleaning",
            "normalization", "standardization", "augmentation", "synthesis", "generation",
            "inference", "prediction", "classification", "regression", "clustering",
            "dimensionality", "reduction", "feature", "extraction", "selection", "engineering"
        ])
        
        return vocab
    
    def generate_text_data(self, num_samples: int = 1000, 
                          min_length: int = 50,
                          max_length: int = 500) -> List[Dict[str, Any]]:
        """Generate synthetic text data"""
        results = []
        
        for i in range(num_samples):
            # Determine text length
            length = random.randint(min_length, max_length)
            
            # Generate text
            words = []
            for _ in range(length):
                word = random.choice(self.vocabulary)
                words.append(word)
                
            # Add some structure
            if random.random() > 0.5:
                # Add a sentence-like structure
                text = " ".join(words).capitalize() + "."
            else:
                # Add paragraph-like structure
                sentences = []
                for _ in range(random.randint(2, 5)):
                    sent_words = random.sample(self.vocabulary, random.randint(8, 20))
                    sentences.append(" ".join(sent_words).capitalize() + ".")
                text = " ".join(sentences)
            
            results.append({
                "text": text,
                "length": len(text),
                "sample_id": i,
                "generated_at": datetime.now().isoformat(),
                "data_type": "text",
                "source": "synthetic_generator"
            })
            
        return results
    
    def generate_qa_pairs(self, num_pairs: int = 500) -> List[Dict[str, Any]]:
        """Generate synthetic question-answer pairs"""
        qa_templates = [
            ("What is {term}?", "{term} is a fundamental concept in {field} that involves {description}."),
            ("How does {term} work?", "{term} works by {mechanism} which enables {functionality}."),
            ("Why is {term} important?", "{term} is important because {importance} which leads to {benefits}."),
            ("When should I use {term}?", "You should use {term} when {scenario} to achieve {goal}."),
            ("What are the benefits of {term}?", "The benefits of {term} include {benefit1}, {benefit2}, and {benefit3}.")
        ]
        
        fields = ["artificial intelligence", "machine learning", "computer science", 
                 "data science", "software engineering", "web development"]
        
        results = []
        
        for i in range(num_pairs):
            term = random.choice(self.vocabulary)
            field = random.choice(fields)
            template = random.choice(qa_templates)
            
            # Fill template
            question = template[0].format(term=term)
            answer = template[1].format(
                term=term,
                field=field,
                description=f"processing {random.choice(['data', 'information', 'signals'])}",
                mechanism=f"using {random.choice(['algorithms', 'models', 'techniques'])}",
                functionality=f"{random.choice(['better performance', 'improved accuracy', 'efficient processing'])}",
                importance=f"it addresses {random.choice(['key challenges', 'fundamental problems', 'critical needs'])}",
                benefits=f"{random.choice(['significant improvements', 'substantial gains', 'major advancements'])}",
                scenario=f"facing {random.choice(['specific requirements', 'particular constraints', 'unique challenges'])}",
                goal=f"{random.choice(['optimal results', 'best performance', 'maximum efficiency'])}",
                benefit1=random.choice(["increased efficiency", "reduced complexity", "improved scalability"]),
                benefit2=random.choice(["better accuracy", "faster processing", "lower costs"]),
                benefit3=random.choice(["enhanced reliability", "greater flexibility", "improved maintainability"])
            )
            
            results.append({
                "question": question,
                "answer": answer,
                "term": term,
                "field": field,
                "pair_id": i,
                "generated_at": datetime.now().isoformat(),
                "data_type": "qa_pair",
                "source": "synthetic_generator"
            })
            
        return results

class DataCollectionManager:
    """Main data collection manager coordinating all collectors"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', 'data/collected')
        self.max_workers = self.config.get('max_workers', 4)
        
        # Initialize collectors
        self.web_collector = WebDataCollector(self.config.get('web', {}))
        self.api_collector = APIDataCollector(self.config.get('api', {}))
        self.synthetic_generator = SyntheticDataGenerator(self.config.get('synthetic', {}))
        
        # Statistics
        self.stats = DataCollectionStats()
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_other=True)
        
    async def collect_web_data(self, urls: List[str]) -> DataCollectionStats:
        """Collect data from web URLs"""
        logger.info(f"Starting web data collection from {len(urls)} URLs")
        
        await self.web_collector.initialize()
        
        try:
            tasks = [self.web_collector.crawl_website(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_data = []
            for result in results:
                if isinstance(result, list):
                    all_data.extend(result)
            
            # Save collected data
            if all_data:
                self._save_data(all_data, "web_data.json")
                self.stats.total_collected += len(all_data)
                self.stats.total_size_bytes += sum(len(json.dumps(item)) for item in all_data)
                self.stats.successful_collections += 1
            
            logger.info(f"Web data collection completed: {len(all_data)} items collected")
            
        finally:
            await self.web_collector.close()
            
        self.stats.sources_processed += len(urls)
        return self.stats
    
    def collect_api_data(self, api_configs: List[Dict[str, Any]]) -> DataCollectionStats:
        """Collect data from APIs"""
        logger.info(f"Starting API data collection from {len(api_configs)} APIs")
        
        for config in api_configs:
            try:
                api_type = config.get('type', '')
                if api_type == 'openai':
                    api_key = config.get('api_key')
                    prompts = config.get('prompts', [])
                    data = self.api_collector.collect_from_openai(api_key, prompts)
                elif api_type == 'huggingface':
                    dataset_name = config.get('dataset_name')
                    split = config.get('split', 'train')
                    data = self.api_collector.collect_from_huggingface(dataset_name, split)
                else:
                    logger.warning(f"Unsupported API type: {api_type}")
                    continue
                
                if data:
                    self._save_data(data, f"{api_type}_data.json")
                    self.stats.total_collected += len(data)
                    self.stats.total_size_bytes += sum(len(json.dumps(item)) for item in data)
                    self.stats.successful_collections += 1
                    
            except Exception as e:
                logger.error(f"Error collecting from API {config.get('type')}: {e}")
                self.stats.failed_collections += 1
        
        self.stats.sources_processed += len(api_configs)
        return self.stats
    
    def generate_synthetic_data(self, config: Dict[str, Any]) -> DataCollectionStats:
        """Generate synthetic data"""
        logger.info("Starting synthetic data generation")
        
        try:
            text_samples = config.get('text_samples', 1000)
            qa_pairs = config.get('qa_pairs', 500)
            
            # Generate text data
            if text_samples > 0:
                text_data = self.synthetic_generator.generate_text_data(text_samples)
                if text_data:
                    self._save_data(text_data, "synthetic_text.json")
                    self.stats.total_collected += len(text_data)
                    self.stats.total_size_bytes += sum(len(json.dumps(item)) for item in text_data)
            
            # Generate QA pairs
            if qa_pairs > 0:
                qa_data = self.synthetic_generator.generate_qa_pairs(qa_pairs)
                if qa_data:
                    self._save_data(qa_data, "synthetic_qa.json")
                    self.stats.total_collected += len(qa_data)
                    self.stats.total_size_bytes += sum(len(json.dumps(item)) for item in qa_data)
            
            self.stats.successful_collections += 1
            logger.info(f"Synthetic data generation completed: {self.stats.total_collected} items generated")
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            self.stats.failed_collections += 1
        
        self.stats.sources_processed += 1
        return self.stats
    
    def _save_data(self, data: List[Dict[str, Any]], filename: str):
        """Save data to file"""
        filepath = os.path.join(self.data_dir, filename)
        
        # Check if file exists and append
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            data = existing_data + data
        
        # Save data
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} items to {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "total_collected": self.stats.total_collected,
            "total_size_mb": round(self.stats.total_size_bytes / (1024 * 1024), 2),
            "sources_processed": self.stats.sources_processed,
            "successful_collections": self.stats.successful_collections,
            "failed_collections": self.stats.failed_collections,
            "duration_seconds": round(self.stats.duration_seconds, 2),
            "collection_rate": round(self.stats.collection_rate, 2)
        }

# Main execution function
def main():
    """Main function for data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Data Collector")
    parser.add_argument("--mode", choices=["web", "api", "synthetic", "all"], 
                       default="synthetic", help="Collection mode")
    parser.add_argument("--output", default="data/collected", help="Output directory")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize manager
    manager = DataCollectionManager({
        'data_dir': args.output,
        'synthetic': {'text_samples': args.samples, 'qa_pairs': args.samples // 2}
    })
    
    if args.mode in ["synthetic", "all"]:
        stats = manager.generate_synthetic_data({
            'text_samples': args.samples,
            'qa_pairs': args.samples // 2
        })
        print(f"Synthetic data generation stats: {stats}")
    
    if args.mode in ["web", "all"]:
        # Example URLs for web collection
        urls = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Deep_learning"
        ]
        
        # Run async web collection
        import asyncio
        stats = asyncio.run(manager.collect_web_data(urls))
        print(f"Web data collection stats: {stats}")
    
    if args.mode in ["api", "all"]:
        # Example API configurations
        api_configs = [
            {
                'type': 'huggingface',
                'dataset_name': 'wikitext',
                'split': 'train',
                'max_samples': 1000
            }
        ]
        
        stats = manager.collect_api_data(api_configs)
        print(f"API data collection stats: {stats}")
    
    # Print final statistics
    final_stats = manager.get_stats()
    print(f"\nFinal collection statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()