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
增强型多语言支持系统 - 统一管理所有语言资源
Enhanced Multilingual Support System - Unified management of all language resources

功能：提供统一的多语言接口，支持代码注释、文档、界面的多语言切换
Function: Provides unified multilingual interface, supports code comments, 
          documentation, and interface language switching
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class EnhancedI18N:
    """增强型多语言管理系统
    Enhanced Multilingual Management System
    
    功能：统一管理代码注释、文档、用户界面的多语言支持
    Function: Unified management of multilingual support for code comments, 
              documentation, and user interfaces
    """
    
    def __init__(self, default_language: str = "zh"):
        """初始化多语言系统
        Initialize multilingual system
        
        Args:
            default_language: 默认语言代码 | Default language code
        """
        self.logger = logging.getLogger(__name__)
        self.default_language = default_language
        self.current_language = default_language
        self.supported_languages = ["zh", "en", "de", "ja", "ru"]
        
        # 加载语言资源 | Load language resources
        self.resources = self._load_language_resources()
        
        # 代码注释模板 | Code comment templates
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
        
        self.logger.info("多语言系统初始化完成 | Multilingual system initialized")

    def _load_language_resources(self) -> Dict[str, Dict[str, str]]:
        """加载所有语言资源文件 | Load all language resource files"""
        resources = {}
        config_dir = Path("config/languages")
        
        for lang in self.supported_languages:
            lang_file = config_dir / f"{lang}.json"
            try:
                if lang_file.exists():
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        resources[lang] = json.load(f)
                else:
                    self.logger.warning(f"语言文件不存在: {lang_file} | Language file not found: {lang_file}")
                    resources[lang] = {}
            except Exception as e:
                self.logger.error(f"加载语言文件失败 {lang}: {str(e)} | Failed to load language file {lang}: {str(e)}")
                resources[lang] = {}
        
        return resources

    def set_language(self, language: str) -> bool:
        """设置当前语言 | Set current language"""
        if language not in self.supported_languages:
            self.logger.warning(f"不支持的语言: {language} | Unsupported language: {language}")
            return False
            
        self.current_language = language
        self.logger.info(f"语言已设置为: {language} | Language set to: {language}")
        return True

    def get_text(self, key: str, default: str = None, **kwargs) -> str:
        """获取翻译文本 | Get translated text"""
        try:
            # 首先尝试当前语言 | First try current language
            if self.current_language in self.resources and key in self.resources[self.current_language]:
                text = self.resources[self.current_language][key]
                return text.format(**kwargs) if kwargs else text
            
            # 然后尝试默认语言 | Then try default language
            if self.default_language in self.resources and key in self.resources[self.default_language]:
                text = self.resources[self.default_language][key]
                return text.format(**kwargs) if kwargs else text
            
            # 最后返回默认值或键名 | Finally return default value or key name
            return default or key
            
        except Exception as e:
            self.logger.error(f"获取文本失败 {key}: {str(e)} | Failed to get text {key}: {str(e)}")
            return default or key

    def get_comment(self, comment_type: str, **kwargs) -> str:
        """获取代码注释模板 | Get code comment template"""
        if comment_type in self.comment_templates and self.current_language in self.comment_templates[comment_type]:
            template = self.comment_templates[comment_type][self.current_language]
            return template.format(**kwargs)
        
        # 回退到英文 | Fallback to English
        if comment_type in self.comment_templates and "en" in self.comment_templates[comment_type]:
            template = self.comment_templates[comment_type]["en"]
            return template.format(**kwargs)
        
        return ""

    def generate_bilingual_comment(self, description: str, params: Dict[str, str] = None, 
                                 returns: str = None) -> str:
        """生成中英文双语注释 | Generate bilingual Chinese-English comments"""
        comments = []
        
        # 功能描述 | Function description
        zh_desc = self.get_comment("function", description=description)
        en_desc = self.comment_templates["function"]["en"].format(description=description)
        comments.append(f"{zh_desc}\n{en_desc}")
        
        # 参数描述 | Parameter descriptions
        if params:
            for param_name, param_desc in params.items():
                zh_param = self.get_comment("param", name=param_name, description=param_desc)
                en_param = self.comment_templates["param"]["en"].format(name=param_name, description=param_desc)
                comments.append(f"{zh_param}\n{en_param}")
        
        # 返回描述 | Return description
        if returns:
            zh_return = self.get_comment("return", description=returns)
            en_return = self.comment_templates["return"]["en"].format(description=returns)
            comments.append(f"{zh_return}\n{en_return}")
        
        return "\n".join(comments)

    def update_resource(self, language: str, key: str, value: str) -> bool:
        """更新语言资源 | Update language resource"""
        if language not in self.supported_languages:
            self.logger.warning(f"不支持的语言: {language} | Unsupported language: {language}")
            return False
        
        if language not in self.resources:
            self.resources[language] = {}
        
        self.resources[language][key] = value
        
        # 保存到文件 | Save to file
        try:
            config_dir = Path("config/languages")
            config_dir.mkdir(exist_ok=True)
            
            lang_file = config_dir / f"{language}.json"
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(self.resources[language], f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"语言资源已更新: {language}/{key} | Language resource updated: {language}/{key}")
            return True
        except Exception as e:
            self.logger.error(f"保存语言资源失败: {str(e)} | Failed to save language resource: {str(e)}")
            return False

    def get_ui_text(self, component: str, element: str) -> str:
        """获取界面文本 | Get UI text"""
        key = f"ui.{component}.{element}"
        return self.get_text(key, f"{component}.{element}")

    def get_all_languages(self) -> Dict[str, Dict[str, str]]:
        """获取所有语言资源 | Get all language resources"""
        return self.resources.copy()

    def export_translations(self, format: str = "json") -> Dict[str, Any]:
        """导出翻译数据 | Export translation data"""
        if format == "json":
            return self.resources
        else:
            self.logger.warning(f"不支持的导出格式: {format} | Unsupported export format: {format}")
            return {}

# 全局实例 | Global instance
i18n = EnhancedI18N()

# 工具函数 | Utility functions
def t(key: str, default: str = None, **kwargs) -> str:
    """翻译快捷函数 | Translation shortcut function"""
    return i18n.get_text(key, default, **kwargs)

def set_app_language(language: str) -> bool:
    """设置应用语言 | Set application language"""
    return i18n.set_language(language)

def get_ui_string(component: str, element: str) -> str:
    """获取界面字符串 | Get UI string"""
    return i18n.get_ui_text(component, element)

# 示例用法 | Example usage
if __name__ == "__main__":
    # 初始化日志 | Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试多语言功能 | Test multilingual functionality
    print("当前语言:", i18n.current_language)
    print("支持的语言:", i18n.supported_languages)
    
    # 测试文本获取 | Test text retrieval
    test_text = i18n.get_text("welcome_message", "欢迎使用Self Soul  | Welcome to AGI System")
    print("欢迎消息:", test_text)
    
    # 测试注释生成 | Test comment generation
    comment = i18n.generate_bilingual_comment(
        description="处理用户输入并生成响应",
        params={"input_data": "输入数据", "context": "上下文信息"},
        returns="处理结果字典"
    )
    print("生成的注释:\n", comment)
    
    # 切换语言测试 | Language switch test
    i18n.set_language("en")
    en_comment = i18n.generate_bilingual_comment(
        description="Process user input and generate response",
        params={"input_data": "Input data", "context": "Context information"},
        returns="Processing result dictionary"
    )
    print("English comment:\n", en_comment)
