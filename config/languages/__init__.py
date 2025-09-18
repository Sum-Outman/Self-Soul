"""
Multi-language support initialization
多语言支持初始化
"""

import os
import json
from typing import Dict, Any

class LanguageManager:
    """Manage multi-language support for the AGI system
       管理Self Soul 的多语言支持
    """
    
    def __init__(self, languages_dir: str = "config/languages"):
        self.languages_dir = languages_dir
        self.supported_languages = ['zh', 'en', 'de', 'ja', 'ru']
        self.current_language = 'zh'
        self.translations: Dict[str, Dict[str, str]] = {}
        
        self._load_all_translations()
    
    def _load_all_translations(self):
        """Load all language translation files
           加载所有语言翻译文件
        """
        for lang_code in self.supported_languages:
            file_path = os.path.join(self.languages_dir, f"{lang_code}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.translations[lang_code] = json.load(f)
                except Exception as e:
                    print(f"Error loading {lang_code} translations: {e}")
                    self.translations[lang_code] = {}
            else:
                print(f"Translation file not found: {file_path}")
                self.translations[lang_code] = {}
    
    def set_language(self, lang_code: str) -> bool:
        """Set current language
           设置当前语言
        """
        if lang_code in self.supported_languages:
            self.current_language = lang_code
            return True
        return False
    
    def get_text(self, key: str, default: str = None, **kwargs) -> str:
        """Get translated text for current language
           获取当前语言的翻译文本
        """
        translation = self.translations.get(self.current_language, {})
        text = translation.get(key, default or key)
        
        # Format with kwargs if provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass
        
        return text
    
    def get_all_translations(self, key: str) -> Dict[str, str]:
        """Get all language translations for a key
           获取一个键的所有语言翻译
        """
        result = {}
        for lang_code in self.supported_languages:
            translation = self.translations.get(lang_code, {})
            result[lang_code] = translation.get(key, key)
        return result

# Global language manager instance
language_manager = LanguageManager()