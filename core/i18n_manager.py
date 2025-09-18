#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一多语言管理系统 - 集成AGI增强的多语言处理能力
Unified Multilingual Management System - Integrated AGI-enhanced multilingual processing

Copyright [2023] [Beijing University of Posts and Telecommunications]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import os
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import defaultdict
import hashlib

# AGI增强导入
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    AGI_CAPABLE = True
except ImportError:
    AGI_CAPABLE = False
    print("警告: AGI增强功能不可用，请安装scikit-learn: pip install scikit-learn")


class I18nManager:
    """统一多语言管理系统 with AGI增强
    Unified Multilingual Management System with AGI Enhancements
    
    功能: 提供智能的多语言支持，包括上下文感知翻译、学习型翻译、代码注释生成
    Function: Provides intelligent multilingual support including context-aware translation, 
              learning-based translation, code comment generation
    """
    
    def __init__(self, default_language: str = "zh"):
        """初始化多语言系统 with AGI能力
        Initialize multilingual system with AGI capabilities
        """
        self.logger = logging.getLogger(__name__)
        self.default_language = default_language
        self.current_language = default_language
        self.supported_languages = ["zh", "en", "de", "ja", "ru"]
        
        # 语言资源存储
        self.resources = {}
        self.translation_memory = defaultdict(dict)  # AGI: 翻译记忆库
        self.context_patterns = defaultdict(list)    # AGI: 上下文模式
        
        # AGI增强组件
        self.vectorizer = None
        self.text_embeddings = {}
        self.learning_enabled = True
        
        # 代码注释模板
        self.comment_templates = {
            "function": {
                "zh": "功能：{description}",
                "en": "Function: {description}",
                "de": "Funktion: {description}",
                "ja": "機能：{description}",
                "ru": "Функция: {description}"
            },
            "param": {
                "zh": "参数：{name} - {description}",
                "en": "Parameter: {name} - {description}",
                "de": "Parameter: {name} - {description}",
                "ja": "パラメータ：{name} - {description}",
                "ru": "Параметр: {name} - {description}"
            },
            "return": {
                "zh": "返回：{description}",
                "en": "Returns: {description}",
                "de": "Rückgabe: {description}",
                "ja": "戻り値：{description}",
                "ru": "Возвращает: {description}"
            }
        }
        
        # 加载语言资源
        self._load_language_resources()
        
        # 初始化AGI组件
        self._init_agi_components()
        
        self.logger.info("AGI增强多语言系统初始化完成 | AGI-enhanced multilingual system initialized")
    
    def _init_agi_components(self):
        """初始化AGI增强组件"""
        if AGI_CAPABLE:
            try:
                self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
                self._train_translation_model()
                self.logger.info("AGI翻译模型初始化成功")
            except Exception as e:
                self.logger.error(f"AGI组件初始化失败: {str(e)}")
                self.learning_enabled = False
        else:
            self.learning_enabled = False
            self.logger.warning("AGI增强功能不可用，使用基础翻译模式")
    
    def _train_translation_model(self):
        """训练翻译模型 - AGI增强"""
        if not AGI_CAPABLE:
            return
            
        # 收集所有翻译文本进行训练
        all_texts = []
        for lang, translations in self.resources.items():
            for key, text in translations.items():
                all_texts.append(text)
                # 为每个文本创建唯一标识
                text_hash = hashlib.md5(text.encode()).hexdigest()
                self.text_embeddings[text_hash] = text
        
        if all_texts:
            try:
                # 训练TF-IDF向量化器
                self.vectorizer.fit(all_texts)
                self.logger.info(f"翻译模型训练完成，共处理 {len(all_texts)} 个文本样本")
            except Exception as e:
                self.logger.error(f"翻译模型训练失败: {str(e)}")
    
    def _load_language_resources(self):
        """加载所有语言资源文件 - 支持多路径资源加载"""
        # 语言资源存储路径（从i18n_unified.py整合）
        resource_paths = [
            Path("config/languages"),
            Path("app/public/languages"),
            Path("app/src/assets/languages"),
            Path("app/src/locales")
        ]
        
        for lang in self.supported_languages:
            self.resources[lang] = {}
            
            # 从所有路径加载资源并合并
            for resource_path in resource_paths:
                lang_file = resource_path / f"{lang}.json"
                try:
                    if lang_file.exists():
                        with open(lang_file, 'r', encoding='utf-8') as f:
                            resources = json.load(f)
                            self._merge_resources(lang, resources, str(lang_file))
                except Exception as e:
                    self.logger.error(f"加载语言文件失败 {lang_file}: {str(e)}")
    
    def _merge_resources(self, lang_code: str, new_resources: Dict, source: str):
        """合并语言资源（从i18n_unified.py整合）"""
        for key, value in new_resources.items():
            if key in self.resources[lang_code]:
                self.logger.warning(f"语言资源键冲突: {key} (来源: {source})")
            self.resources[lang_code][key] = value
            
        # 计算校验和用于变化检测
        content_hash = hashlib.md5(json.dumps(new_resources, sort_keys=True).encode()).hexdigest()
        # 存储校验和信息（可根据需要扩展）
    
    def set_language(self, language: str) -> bool:
        """设置当前语言"""
        if language not in self.supported_languages:
            self.logger.warning(f"不支持的语言: {language}")
            return False
            
        self.current_language = language
        self.logger.info(f"语言已设置为: {language}")
        return True
    
    # AGI增强翻译方法
    def translate_with_context(self, text: str, context: str = None, similarity_threshold: float = 0.7) -> str:
        """上下文感知翻译 - AGI增强
        Context-aware translation with AGI enhancement
        """
        # 首先尝试精确匹配
        exact_match = self.get_text(text, None)
        if exact_match != text:
            return exact_match
        
        # AGI: 使用上下文信息进行智能翻译
        if context and AGI_CAPABLE and self.learning_enabled:
            try:
                # 基于上下文查找相似翻译
                context_key = f"{context}_{text}"
                if context_key in self.translation_memory:
                    return self.translation_memory[context_key].get(self.current_language, text)
                
                # 使用机器学习查找相似文本
                similar_translation = self._find_similar_translation(text, context, similarity_threshold)
                if similar_translation:
                    # 学习并存储这个翻译
                    self._learn_translation(text, similar_translation, context)
                    return similar_translation
            except Exception as e:
                self.logger.error(f"上下文翻译失败: {str(e)}")
        
        # 回退到基础翻译
        return self.get_text(text, text)
    
    def _find_similar_translation(self, text: str, context: str, threshold: float = 0.7) -> Optional[str]:
        """使用机器学习查找相似翻译"""
        if not AGI_CAPABLE or not self.vectorizer:
            return None
            
        try:
            # 将查询文本向量化
            query_vec = self.vectorizer.transform([text])
            
            # 获取当前语言的所有文本进行相似度计算
            lang_texts = list(self.resources[self.current_language].values())
            if not lang_texts:
                return None
                
            # 向量化目标语言文本
            target_vecs = self.vectorizer.transform(lang_texts)
            
            # 计算相似度
            similarities = cosine_similarity(query_vec, target_vecs)[0]
            
            # 找到最相似的文本
            max_similarity = np.max(similarities)
            if max_similarity >= threshold:
                max_index = np.argmax(similarities)
                return lang_texts[max_index]
                
        except Exception as e:
            self.logger.error(f"相似翻译查找失败: {str(e)}")
            
        return None
    
    def _learn_translation(self, source_text: str, target_text: str, context: str = None):
        """学习新的翻译对 - AGI增强"""
        if not self.learning_enabled:
            return
            
        # 存储到翻译记忆库
        if context:
            context_key = f"{context}_{source_text}"
            self.translation_memory[context_key][self.current_language] = target_text
        
        # 添加到资源中以便持久化
        key = f"learned_{hashlib.md5(source_text.encode()).hexdigest()[:8]}"
        self.resources[self.current_language][key] = target_text
        
        self.logger.info(f"学习新翻译: '{source_text}' -> '{target_text}'")
    
    def get_text(self, key: str, default: str = None, **kwargs) -> str:
        """获取翻译文本"""
        try:
            # 首先尝试当前语言
            if self.current_language in self.resources and key in self.resources[self.current_language]:
                text = self.resources[self.current_language][key]
                return text.format(**kwargs) if kwargs else text
            
            # 然后尝试默认语言
            if self.default_language in self.resources and key in self.resources[self.default_language]:
                text = self.resources[self.default_language][key]
                return text.format(**kwargs) if kwargs else text
            
            # 最后返回默认值或键名
            return default or key
            
        except Exception as e:
            self.logger.error(f"获取文本失败 {key}: {str(e)}")
            return default or key
    
    def get_comment(self, comment_type: str, **kwargs) -> str:
        """获取代码注释模板"""
        if comment_type in self.comment_templates and self.current_language in self.comment_templates[comment_type]:
            template = self.comment_templates[comment_type][self.current_language]
            return template.format(**kwargs)
        
        # 回退到英文
        if comment_type in self.comment_templates and "en" in self.comment_templates[comment_type]:
            template = self.comment_templates[comment_type]["en"]
            return template.format(**kwargs)
        
        return ""
    
    def generate_bilingual_comment(self, description: str, params: Dict[str, str] = None, 
                                 returns: str = None) -> str:
        """生成中英文双语注释"""
        comments = []
        
        # 功能描述
        zh_desc = self.get_comment("function", description=description)
        en_desc = self.comment_templates["function"]["en"].format(description=description)
        comments.append(f"{zh_desc}\n{en_desc}")
        
        # 参数描述
        if params:
            for param_name, param_desc in params.items():
                zh_param = self.get_comment("param", name=param_name, description=param_desc)
                en_param = self.comment_templates["param"]["en"].format(name=param_name, description=param_desc)
                comments.append(f"{zh_param}\n{en_param}")
        
        # 返回描述
        if returns:
            zh_return = self.get_comment("return", description=returns)
            en_return = self.comment_templates["return"]["en"].format(description=returns)
            comments.append(f"{zh_return}\n{en_return}")
        
        return "\n".join(comments)
    
    def update_resource(self, language: str, key: str, value: str) -> bool:
        """更新语言资源"""
        if language not in self.supported_languages:
            self.logger.warning(f"不支持的语言: {language}")
            return False
        
        if language not in self.resources:
            self.resources[language] = {}
        
        self.resources[language][key] = value
        
        # 保存到文件
        try:
            config_dir = Path("config/languages")
            config_dir.mkdir(exist_ok=True)
            
            lang_file = config_dir / f"{language}.json"
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(self.resources[language], f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"语言资源已更新: {language}/{key}")
            return True
        except Exception as e:
            self.logger.error(f"保存语言资源失败: {str(e)}")
            return False
    
    def get_ui_text(self, component: str, element: str) -> str:
        """获取界面文本"""
        key = f"ui.{component}.{element}"
        return self.get_text(key, f"{component}.{element}")
    
    def get_all_languages(self) -> Dict[str, Dict[str, str]]:
        """获取所有语言资源"""
        return self.resources.copy()
    
    def export_translations(self, format: str = "json") -> Dict[str, Any]:
        """导出翻译数据"""
        if format == "json":
            return self.resources
        else:
            self.logger.warning(f"不支持的导出格式: {format}")
            return {}
    
    # AGI增强方法
    def enable_learning(self, enabled: bool = True):
        """启用或禁用学习功能"""
        self.learning_enabled = enabled and AGI_CAPABLE
        self.logger.info(f"学习功能已{'启用' if self.learning_enabled else '禁用'}")
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """获取翻译统计信息 - AGI增强"""
        stats = {
            "total_translations": sum(len(lang_dict) for lang_dict in self.resources.values()),
            "languages": list(self.resources.keys()),
            "translation_memory_size": len(self.translation_memory),
            "learning_enabled": self.learning_enabled,
            "agi_capable": AGI_CAPABLE
        }
        
        for lang in self.supported_languages:
            stats[f"{lang}_count"] = len(self.resources.get(lang, {}))
            
        return stats

# 全局实例
i18n_manager = I18nManager()

# 向后兼容的函数
def translate_text(text: str, target_language: str = None) -> str:
    """翻译文本到指定语言（兼容旧版本）"""
    if target_language:
        original_lang = i18n_manager.current_language
        i18n_manager.set_language(target_language)
        result = i18n_manager.get_text(text, text)
        i18n_manager.set_language(original_lang)
        return result
    else:
        return i18n_manager.get_text(text, text)

def gettext(text: str) -> str:
    """获取翻译文本的模块级函数（兼容旧版本）"""
    return i18n_manager.get_text(text, text)

def _(text: str) -> str:
    """获取翻译的便捷函数（兼容旧版本）"""
    return gettext(text)

def set_language(language: str):
    """设置全局语言的模块级函数（兼容旧版本）"""
    i18n_manager.set_language(language)

def get_supported_languages() -> list:
    """获取所有支持的语言（兼容旧版本）"""
    return i18n_manager.supported_languages

# 新API函数
def t(key: str, default: str = None, **kwargs) -> str:
    """翻译快捷函数"""
    return i18n_manager.get_text(key, default, **kwargs)

def translate_with_context(text: str, context: str = None) -> str:
    """上下文感知翻译"""
    return i18n_manager.translate_with_context(text, context)

# 示例用法
if __name__ == "__main__":
    # 初始化日志
    logging.basicConfig(level=logging.INFO)
    
    # 测试多语言功能
    print("当前语言:", i18n_manager.current_language)
    print("支持的语言:", i18n_manager.supported_languages)
    
    # 测试文本获取
    test_text = i18n_manager.get_text("welcome_message", "欢迎使用Self Soul  | Welcome to AGI System")
    print("欢迎消息:", test_text)
    
    # 测试AGI增强功能
    stats = i18n_manager.get_translation_stats()
    print("翻译统计:", stats)
    
    # 测试上下文翻译
    context_text = translate_with_context("processing", "computer_vision")
    print("上下文翻译:", context_text)
