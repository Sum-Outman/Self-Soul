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
统一语言资源管理系统 - 集中管理所有语言资源
Unified Language Resource Management System - Centralized management of all language resources

功能：提供统一的语言资源访问接口，确保所有组件使用一致的语言数据
Function: Provides unified language resource access interface, ensuring all components use consistent language data
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

class UnifiedLanguageManager:
    """统一语言资源管理器
    Unified Language Resource Manager
    
    功能：集中管理所有语言资源，提供统一的访问接口和多语言支持
    Function: Centralized management of all language resources, providing unified access interface and multilingual support
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_languages = {
            "zh": {"name": "简体中文", "code": "zh", "enabled": True},
            "en": {"name": "English", "code": "en", "enabled": True},
            "de": {"name": "Deutsch", "code": "de", "enabled": True},
            "ja": {"name": "日本語", "code": "ja", "enabled": True},
            "ru": {"name": "Русский", "code": "ru", "enabled": True}
        }
        
        # 语言资源存储路径
        self.resource_paths = [
            Path("config/languages"),
            Path("app/public/languages"),
            Path("app/src/assets/languages"),
            Path("app/src/locales")
        ]
        
        # 统一资源存储
        self.unified_resources = {}
        self.resource_checksums = {}
        
        self._load_all_resources()
        
    def _load_all_resources(self):
        """加载所有语言资源文件 | Load all language resource files"""
        for lang_code in self.supported_languages:
            self.unified_resources[lang_code] = {}
            self._load_language_resources(lang_code)
    
    def _load_language_resources(self, lang_code: str):
        """加载指定语言的所有资源 | Load all resources for specified language"""
        for resource_path in self.resource_paths:
            lang_file = resource_path / f"{lang_code}.json"
            if lang_file.exists():
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        resources = json.load(f)
                        self._merge_resources(lang_code, resources, str(lang_file))
                except Exception as e:
                    self.logger.error(f"加载语言文件失败 {lang_file}: {str(e)} | Failed to load language file {lang_file}: {str(e)}")
    
    def _merge_resources(self, lang_code: str, new_resources: Dict, source: str):
        """合并语言资源 | Merge language resources"""
        for key, value in new_resources.items():
            if key in self.unified_resources[lang_code]:
                self.logger.warning(f"语言资源键冲突: {key} (来源: {source}) | Language resource key conflict: {key} (source: {source})")
            self.unified_resources[lang_code][key] = value
            
        # 计算校验和用于变化检测
        content_hash = hashlib.md5(json.dumps(new_resources, sort_keys=True).encode()).hexdigest()
        self.resource_checksums[f"{lang_code}_{source}"] = content_hash
    
    def get_text(self, lang_code: str, key: str, default: str = None, **kwargs) -> str:
        """获取翻译文本 | Get translated text"""
        if lang_code not in self.supported_languages:
            self.logger.warning(f"不支持的语言代码: {lang_code} | Unsupported language code: {lang_code}")
            lang_code = "en"  # 默认回退到英文
        
        if lang_code in self.unified_resources and key in self.unified_resources[lang_code]:
            text = self.unified_resources[lang_code][key]
            try:
                return text.format(**kwargs) if kwargs else text
            except KeyError as e:
                self.logger.error(f"文本格式化错误 {key}: {str(e)} | Text formatting error {key}: {str(e)}")
                return text
        
        return default or key
    
    def set_language(self, lang_code: str) -> bool:
        """设置当前语言 | Set current language"""
        if lang_code not in self.supported_languages:
            self.logger.warning(f"不支持的语言: {lang_code} | Unsupported language: {lang_code}")
            return False
        
        self.current_language = lang_code
        self.logger.info(f"语言已设置为: {self.supported_languages[lang_code]['name']} | Language set to: {self.supported_languages[lang_code]['name']}")
        return True
    
    def update_resource(self, lang_code: str, key: str, value: str, persist: bool = True) -> bool:
        """更新语言资源 | Update language resource"""
        if lang_code not in self.supported_languages:
            return False
        
        if lang_code not in self.unified_resources:
            self.unified_resources[lang_code] = {}
        
        self.unified_resources[lang_code][key] = value
        
        if persist:
            # 保存到主配置文件
            main_config_path = Path("config/languages") / f"{lang_code}.json"
            self._save_resources(lang_code, main_config_path)
        
        return True
    
    def _save_resources(self, lang_code: str, file_path: Path):
        """保存语言资源到文件 | Save language resources to file"""
        try:
            file_path.parent.mkdir(exist_ok=True, parents=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.unified_resources[lang_code], f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存语言资源失败 {file_path}: {str(e)} | Failed to save language resource {file_path}: {str(e)}")
    
    def export_translations(self, format: str = "json") -> Dict[str, Any]:
        """导出所有翻译数据 | Export all translation data"""
        if format == "json":
            return self.unified_resources
        else:
            self.logger.warning(f"不支持的导出格式: {format} | Unsupported export format: {format}")
            return {}
    
    def get_ui_text(self, component: str, element: str, lang_code: str = None) -> str:
        """获取界面文本 | Get UI text"""
        if lang_code is None:
            lang_code = getattr(self, 'current_language', 'en')
        
        key = f"ui.{component}.{element}"
        return self.get_text(lang_code, key, f"{component}.{element}")
    
    def generate_bilingual_docstring(self, zh_description: str, en_description: str) -> str:
        """生成中英文双语文档字符串 | Generate bilingual Chinese-English docstring"""
        return f'"""{zh_description}\n\n{en_description}"""'

# 全局实例 | Global instance
language_manager = UnifiedLanguageManager()

# 工具函数 | Utility functions
def t(key: str, lang_code: str = None, default: str = None, **kwargs) -> str:
    """翻译快捷函数 | Translation shortcut function"""
    if lang_code is None:
        lang_code = getattr(language_manager, 'current_language', 'en')
    return language_manager.get_text(lang_code, key, default, **kwargs)

def set_app_language(lang_code: str) -> bool:
    """设置应用语言 | Set application language"""
    return language_manager.set_language(lang_code)

def get_ui_string(component: str, element: str, lang_code: str = None) -> str:
    """获取界面字符串 | Get UI string"""
    return language_manager.get_ui_text(component, element, lang_code)

# 示例用法 | Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试统一语言管理器
    print("支持的语言:", language_manager.supported_languages.keys())
    
    # 测试文本获取
    test_text = t("welcome_message", "zh", "欢迎使用AGI系统")
    print("中文欢迎消息:", test_text)
    
    test_text = t("welcome_message", "en", "Welcome to AGI System")
    print("英文欢迎消息:", test_text)
    
    # 测试文档字符串生成
    docstring = language_manager.generate_bilingual_docstring(
        "处理用户输入并生成响应",
        "Process user input and generate response"
    )
    print("生成的文档字符串:\n", docstring)