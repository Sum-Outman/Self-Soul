#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
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

"""
多语言支持模块 - 提供中英文翻译功能
Multilingual support module - Provides Chinese and English translation functionality
"""
import json
import os
from typing import Dict, Any


class I18n:
    """国际化翻译类
    Internationalization translation class
    """
    
    def __init__(self, locale_dir: str = None):
        """初始化多语言支持
        Initialize multilingual support
        
        Args:
            locale_dir: 语言文件目录路径 | Path to language files directory
        """
        self.locale_dir = locale_dir or os.path.join(os.path.dirname(__file__), '..', 'config', 'languages')
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language = 'zh'  # 默认中文 Default Chinese
        
        # 加载所有语言文件 Load all language files
        self._load_translations()
    
    
    def _load_translations(self):
        """加载所有可用的语言翻译
        Load all available language translations
        """
        try:
            if os.path.exists(self.locale_dir):
                for filename in os.listdir(self.locale_dir):
                    if filename.endswith('.json'):
                        lang_code = filename.split('.')[0]
                        filepath = os.path.join(self.locale_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            self.translations[lang_code] = json.load(f)
        except Exception as e:
            print(f"加载语言文件失败: {e} | Failed to load language files: {e}")
            # 使用空翻译作为回退 Use empty translations as fallback
            self.translations = {'zh': {}, 'en': {}}
    
    
    def set_language(self, language: str):
        """设置当前语言
        Set current language
        
        Args:
            language: 语言代码 (zh, en, de, ja, ru) | Language code (zh, en, de, ja, ru)
        """
        if language in self.translations:
            self.current_language = language
        else:
            print(f"不支持的语言: {language} | Unsupported language: {language}")
    
    
    def gettext(self, text: str) -> str:
        """获取翻译文本
        Get translated text
        
        Args:
            text: 要翻译的文本 | Text to translate
            
        Returns:
            翻译后的文本 | Translated text
        """
        # 如果在当前语言中找到翻译，返回翻译文本
        # If translation found in current language, return translated text
        if (self.current_language in self.translations and 
            text in self.translations[self.current_language]):
            return self.translations[self.current_language][text]
        
        # 如果在英文中找到翻译，返回英文翻译
        # If translation found in English, return English translation
        if ('en' in self.translations and text in self.translations['en']):
            return self.translations['en'][text]
        
        # 如果没有找到翻译，返回原文
        # If no translation found, return original text
        return text


# 创建全局的I18n实例
_i18n_instance = I18n()


def translate_text(text: str, target_language: str = None) -> str:
    """翻译文本到指定语言
    Translate text to target language
    
    Args:
        text: 要翻译的文本 | Text to translate
        target_language: 目标语言代码 | Target language code
        
    Returns:
        翻译后的文本 | Translated text
    """
    if target_language and target_language in _i18n_instance.translations:
        original_language = _i18n_instance.current_language
        try:
            _i18n_instance.set_language(target_language)
            return _i18n_instance.gettext(text)
        finally:
            # 恢复原来的语言设置
            _i18n_instance.set_language(original_language)
    else:
        # 使用当前语言或默认语言
        return _i18n_instance.gettext(text)


def gettext(text: str) -> str:
    """获取翻译文本的模块级函数
    Module-level function to get translated text
    
    Args:
        text: 要翻译的文本 | Text to translate
        
    Returns:
        翻译后的文本 | Translated text
    """
    return _i18n_instance.gettext(text)


def _(text: str) -> str:
    """获取翻译的便捷函数
    Convenient function to get translation
    
    Args:
        text: 要翻译的文本 | Text to translate
        
    Returns:
        翻译后的文本 | Translated text
    """
    return gettext(text)


def set_language(language: str):
    """设置全局语言的模块级函数
    Module-level function to set global language
    
    Args:
        language: 语言代码 | Language code
    """
    _i18n_instance.set_language(language)


def get_supported_languages() -> list:
    """获取所有支持的语言
    Get all supported languages
    
    Returns:
        支持的语言代码列表 | List of supported language codes
    """
    return list(_i18n_instance.translations.keys())
