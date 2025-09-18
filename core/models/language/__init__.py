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
语言模型包初始化文件
Language Model Package Initialization File

导出语言模型类以供其他模块使用
Export language model class for use by other modules
"""

from .model import LanguageModel, AdvancedLanguageModel

__all__ = ['LanguageModel', 'AdvancedLanguageModel']
