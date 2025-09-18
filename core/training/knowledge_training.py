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
基础模型类 - 所有模型的基类
Base Model Class - Base class for all models

提供通用接口和功能，确保所有模型的一致性
Provides common interfaces and functionality to ensure consistency across all models
"""
"""
知识库模型训练程序
Knowledge Model Training Program
"""

import os
import json
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import logging
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
KnowledgeDataset类 - 中文类描述
KnowledgeDataset Class - English class description
"""
class KnowledgeDataset(Dataset):
    """知识库模型训练数据集 | Knowledge model training dataset"""
    
"""
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self, data_dir, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.knowledge_graph = nx.DiGraph()
        
        # 加载多学科知识数据
        disciplines = [
            'physics', 'mathematics', 'chemistry', 'biology', 
            'computer_science', 'engineering', 'medicine', 'law',
            'economics', 'psychology', 'history', 'philosophy'
        ]
        
        for discipline in disciplines:
            file_path = os.path.join(data_dir, f"{discipline}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    discipline_data = json.load(f)
                    self.data.extend(discipline_data)
                    
                    # 构建知识图谱
                    for concept in discipline_data:
                        self.knowledge_graph.add_node(concept['concept'], type='concept', domain=discipline)
                        for relation in concept['relations']:
                            self.knowledge_graph.add_edge(
                                concept['concept'], 
                                relation['target'], 
                                relation=relation['type']
                            )
        
        logger.info(f"知识图谱构建完成，包含 {len(self.knowledge_graph.nodes)} 个节点和 {len(self.knowledge_graph.edges)} 条边 | "
                    f"Knowledge graph built with {len(self.knowledge_graph.nodes)} nodes and {len(self.knowledge_graph.edges)} edges")
    
    
"""
__len__函数 - 中文函数描述
__len__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __len__(self):
        return len(self.data)
    
    
"""
__getitem__函数 - 中文函数描述
__getitem__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本
        context = f"[CONCEPT]{item['concept']}[/CONCEPT] [DEFINITION]{item['definition']}[/DEFINITION]"
        
        # 添加相关概念
        if 'relations' in item:
            for rel in item['relations']:
                context += f" [RELATION]{rel['type']}:{rel['target']}[/RELATION]"
        
        # 标记化处理
        encoding = self.tokenizer(
            context,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'concept': item['concept'],
            'definition': item['definition']
        }


"""
KnowledgeModelTrainer类 - 中文类描述
KnowledgeModelTrainer Class - English class description
"""
class KnowledgeModelTrainer:
    
    """
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化BERT模型
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        # 知识嵌入层
        self.knowledge_embedding = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512)
        ).to(self.device)
        
        # 跨领域融合层
        self.cross_domain_fusion = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048
        ).to(self.device)
        
        # 损失函数
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        
    
"""
load_data函数 - 中文函数描述
load_data Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def load_data(self, data_dir):
        """加载知识库数据 | Load knowledge data"""
        dataset = KnowledgeDataset(data_dir, self.tokenizer)
        return dataset
    
    
"""
create_data_loader函数 - 中文函数描述
create_data_loader Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def create_data_loader(self, dataset, batch_size=8, shuffle=True):
        """创建数据加载器 | Create data loader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4
        )
    
    
"""
train_epoch函数 - 中文函数描述
train_epoch Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def train_epoch(self, train_loader, optimizer):
        """单轮训练 | Single epoch training"""
        self.bert.train()
        self.knowledge_embedding.train()
        self.cross_domain_fusion.train()
        
        total_loss = 0
        for batch in tqdm(train_loader, desc="训练中 | Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            optimizer.zero_grad()
            
            # BERT前向传播
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            
            # 知识嵌入
            knowledge_emb = self.knowledge_embedding(pooled_output)
            
            # 跨领域融合
            fused_knowledge = self.cross_domain_fusion(knowledge_emb.unsqueeze(0)).squeeze(0)
            
            # 三元组损失（锚点、正例、负例）
            anchor = fused_knowledge[0]
            positive = fused_knowledge[1]
            negative = fused_knowledge[2]
            
            loss = self.triplet_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    
"""
evaluate函数 - 中文函数描述
evaluate Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def evaluate(self, test_loader):
        """模型评估 | Model evaluation"""
        self.bert.eval()
        self.knowledge_embedding.eval()
        self.cross_domain_fusion.eval()
        
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中 | Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # BERT前向传播
                outputs = self.bert(input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                
                # 知识嵌入
                knowledge_emb = self.knowledge_embedding(pooled_output)
                
                # 跨领域融合
                fused_knowledge = self.cross_domain_fusion(knowledge_emb.unsqueeze(0)).squeeze(0)
                
                # 三元组损失
                anchor = fused_knowledge[0]
                positive = fused_knowledge[1]
                negative = fused_knowledge[2]
                
                loss = self.triplet_loss(anchor, positive, negative)
                total_loss += loss.item()
                
        return total_loss / len(test_loader)
    
    
"""
save_model函数 - 中文函数描述
save_model Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def save_model(self, path):
        """保存模型 | Save model"""
        torch.save({
            'bert_state_dict': self.bert.state_dict(),
            'knowledge_embedding_state_dict': self.knowledge_embedding.state_dict(),
            'cross_domain_fusion_state_dict': self.cross_domain_fusion.state_dict()
        }, path)
        logger.info(f"知识模型已保存至 {path} | Knowledge model saved to {path}")
    
    
"""
full_training函数 - 中文函数描述
full_training Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def full_training(self, data_dir, epochs=10):
        """完整训练流程 | Full training pipeline"""
        # 加载数据
        dataset = self.load_data(data_dir)
        
        # 划分训练集和测试集
        train_size = int(0.85 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # 创建数据加载器
        train_loader = self.create_data_loader(train_dataset, batch_size=self.config['batch_size'])
        test_loader = self.create_data_loader(test_dataset, batch_size=self.config['batch_size'])
        
        # 设置优化器
        optimizer = optim.AdamW([
            {'params': self.bert.parameters()},
            {'params': self.knowledge_embedding.parameters()},
            {'params': self.cross_domain_fusion.parameters()}
        ], lr=self.config['learning_rate'])
        
        # 训练循环
        best_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            test_loss = self.evaluate(test_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"训练损失: {train_loss:.4f} | 测试损失: {test_loss:.4f} | "
                f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
            )
            
            # 保存最佳模型
            if test_loss < best_loss:
                best_loss = test_loss
                self.save_model(self.config['model_save_path'])
        
        logger.info("知识库模型训练完成 | Knowledge model training completed")

# 配置示例 | Example configuration
if __name__ == "__main__":
    config = {
        'batch_size': 8,
        'learning_rate': 2e-5,
        'model_save_path': 'models/knowledge_model.pth'
    }
    
    # 知识库数据目录
    knowledge_data_dir = 'data/knowledge/'
    
    # 创建数据目录（如果不存在）
    os.makedirs(knowledge_data_dir, exist_ok=True)
    
    trainer = KnowledgeModelTrainer(config)
    trainer.full_training(knowledge_data_dir, epochs=15)
