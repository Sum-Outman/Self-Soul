#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试FromScratchTrainer类的增强功能

This script tests the enhanced functionality of the FromScratchTrainer class,
including improved training with batch processing and learning rate decay,
more sophisticated text generation, sentiment analysis, and summarization.
"""

import logging
import sys
import os
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导入FromScratchTrainer类
from core.models.language.model import FromScratchTrainer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FromScratchTrainerTest")

class FromScratchTrainerTester:
    """测试FromScratchTrainer类的功能"""
    
    def __init__(self):
        # 创建训练器实例，配置增强功能
        self.config = {
            'embedding_dim': 100,
            'window_size': 2,
            'min_count': 1,
            'learning_rate': 0.01,
            'epochs': 10,
            'hidden_size': 128,
            'sequence_length': 10,
            'batch_size': 32,  # 新增：批处理大小
            'decay_rate': 0.9,  # 新增：学习率衰减率
            'decay_steps': 3    # 新增：学习率衰减步数
        }
        
        self.trainer = FromScratchTrainer(self.config)
        self.test_data = self._prepare_test_data()
        
    def _prepare_test_data(self) -> List[str]:
        """准备测试数据"""
        # 简单的英语句子作为训练数据
        test_sentences = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is fascinating",
            "I love programming and creating new things",
            "The weather is beautiful today",
            "I am happy to see you",
            "This is a great day for learning",
            "I feel excited about the new project",
            "Programming is challenging but rewarding",
            "The sun is shining brightly in the sky",
            "I enjoy reading books and learning new concepts",
            "The food tastes delicious and satisfying",
            "I am disappointed with the result",
            "This is a terrible mistake that needs to be fixed",
            "The experience was wonderful and memorable",
            "I hate it when things don't work as expected"
        ]
        
        # 复制几次以增加数据量
        return test_sentences * 5
    
    def test_training(self) -> bool:
        """测试训练功能"""
        try:
            logger.info("Starting training test...")
            self.trainer.train(self.test_data)
            logger.info("Training test completed successfully")
            
            # 验证模型是否正确初始化
            assert self.trainer.embeddings is not None, "Embeddings not initialized after training"
            assert self.trainer.vocab_size > 0, "Vocabulary size is zero after training"
            
            return True
        except Exception as e:
            logger.error(f"Training test failed: {str(e)}")
            return False
    
    def test_text_generation(self) -> bool:
        """测试文本生成功能"""
        try:
            logger.info("Starting text generation test...")
            
            # 测试不同的种子文本
            seed_texts = [
                "The quick",
                "Machine learning",
                "I love",
                "The weather",
                ""
            ]
            
            for seed_text in seed_texts:
                logger.info(f"Generating text with seed: '{seed_text}'")
                generated_text = self.trainer.generate_text(seed_text, max_length=50)
                logger.info(f"Generated text: '{generated_text}'")
                
                # 验证生成的文本不为空
                assert generated_text.strip() != "", "Generated text is empty"
                
                # 验证生成的文本不包含特殊标记
                for special_token in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
                    assert special_token not in generated_text, f"Generated text contains special token: {special_token}"
            
            logger.info("Text generation test completed successfully")
            return True
        except Exception as e:
            logger.error(f"Text generation test failed: {str(e)}")
            return False
    
    def test_sentiment_analysis(self) -> bool:
        """测试情感分析功能"""
        try:
            logger.info("Starting sentiment analysis test...")
            
            # 测试不同情感的句子
            test_sentences = [
                ("I am very happy and excited about this great opportunity", "positive"),
                ("This is a terrible mistake and I am very disappointed", "negative"),
                ("The meeting will start at 3 PM tomorrow", "neutral"),
                ("I do not like this bad product", "negative"),  # 测试否定词
                ("I am not unhappy with the result", "positive"),  # 测试双重否定
                ("", "neutral")  # 测试空文本
            ]
            
            for text, expected_sentiment in test_sentences:
                logger.info(f"Analyzing sentiment for: '{text}'")
                result = self.trainer.analyze_sentiment(text)
                logger.info(f"Sentiment result: {result}")
                
                # 验证结果格式正确
                assert all(key in result for key in ["positive", "negative", "neutral"]), "Invalid sentiment result format"
                
                # 验证分数和为1
                total_score = sum(result.values())
                assert abs(total_score - 1.0) < 0.01, f"Sentiment scores do not sum to 1: {total_score}"
                
                # 检查主导情感（除了空文本）
                if text:
                    dominant_emotion = max(result, key=result.get)
                    if expected_sentiment == "positive":
                        assert result["positive"] > result["negative"], f"Expected positive sentiment, got {dominant_emotion}"
                    elif expected_sentiment == "negative":
                        assert result["negative"] > result["positive"], f"Expected negative sentiment, got {dominant_emotion}"
                    elif expected_sentiment == "neutral":
                        assert result["neutral"] > result["positive"] and result["neutral"] > result["negative"], \
                            f"Expected neutral sentiment, got {dominant_emotion}"
            
            logger.info("Sentiment analysis test completed successfully")
            return True
        except Exception as e:
            logger.error(f"Sentiment analysis test failed: {str(e)}")
            return False
    
    def test_text_summarization(self) -> bool:
        """测试文本摘要功能"""
        try:
            logger.info("Starting text summarization test...")
            
            # 测试不同类型的文本
            test_texts = [
                "This is a simple test sentence. It contains important information. We want to see if the summarizer works correctly.",
                "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
                "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
                "",  # 测试空文本
                "Short text"  # 测试短文本
            ]
            
            for text in test_texts:
                max_length = min(50, len(text) // 2) if len(text) > 0 else 0
                logger.info(f"Summarizing text (max length: {max_length}): '{text}'")
                summary = self.trainer.summarize_text(text, max_length)
                logger.info(f"Summary: '{summary}'")
                
                # 验证摘要长度不超过最大长度
                if max_length > 0:
                    assert len(summary) <= max_length + 3, f"Summary too long: {len(summary)} > {max_length}"  # +3 for ellipsis
                
                # 验证空文本返回空字符串
                if not text:
                    assert summary == "", "Empty text should return empty summary"
                
                # 验证短文本直接返回
                if len(text) <= max_length and text:
                    assert summary == text, "Short text should be returned as is"
            
            logger.info("Text summarization test completed successfully")
            return True
        except Exception as e:
            logger.error(f"Text summarization test failed: {str(e)}")
            return False
    
    def test_language_detection(self) -> bool:
        """测试语言检测功能"""
        try:
            logger.info("Starting language detection test...")
            
            # 测试不同语言的文本
            test_texts = [
                "This is an English sentence",
                "Ceci est une phrase en français",  # French
                "Dies ist ein deutscher Satz",  # German
                "这是一个中文句子",  # Chinese
                ""  # 测试空文本
            ]
            
            for text in test_texts:
                logger.info(f"Detecting language for: '{text}'")
                result = self.trainer.detect_language(text)
                logger.info(f"Language detection result: {result}")
                
                # 验证结果格式正确
                assert all(key in result for key in ["language", "confidence"]), "Invalid language detection result format"
                
                # 根据系统要求，语言应该始终是英语
                assert result["language"] == "en", f"Language should be 'en', got '{result['language']}'"
                
                # 验证置信度在合理范围内
                assert 0.0 <= result["confidence"] <= 1.0, f"Confidence should be between 0 and 1, got {result['confidence']}"
            
            logger.info("Language detection test completed successfully")
            return True
        except Exception as e:
            logger.error(f"Language detection test failed: {str(e)}")
            return False
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        logger.info("Running all FromScratchTrainer tests...")
        
        tests = [
            ("Training", self.test_training),
            ("Text Generation", self.test_text_generation),
            ("Sentiment Analysis", self.test_sentiment_analysis),
            ("Text Summarization", self.test_text_summarization),
            ("Language Detection", self.test_language_detection)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            logger.info(f"Running {test_name} test...")
            if not test_func():
                all_passed = False
                logger.error(f"{test_name} test FAILED")
            else:
                logger.info(f"{test_name} test PASSED")
        
        if all_passed:
            logger.info("All tests PASSED!")
        else:
            logger.error("Some tests FAILED!")
        
        return all_passed

if __name__ == "__main__":
    tester = FromScratchTrainerTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)