"""
自动添加中英文注释工具
Automated Bilingual Comments Tool

功能：自动为Python文件添加中英文对照注释
Function: Automatically add bilingual comments to Python files
"""

import os
import re
import argparse
from pathlib import Path

def add_bilingual_comments(file_path):
    """为文件添加中英文注释 | Add bilingual comments to file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加文件头注释 | Add file header comments
    header_pattern = r'("""|\'\'\')\s*\* Licensed under the Apache License'
    if not re.search(header_pattern, content):
        header = '''"""
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
'''
        content = header + content
    
    # 添加类注释模板 | Add class comment template
    class_pattern = r'class\s+(\w+)(?:\(|:)'
    classes = re.findall(class_pattern, content)
    
    for class_name in classes:
        class_comment_pattern = fr'class\s+{class_name}(?:\(|:)(.*?)(?:"""|\'\'\')'
        if not re.search(class_comment_pattern, content, re.DOTALL):
            class_comment = f'''
"""
{class_name}类 - 中文类描述
{class_name} Class - English class description

功能：详细描述类的功能
Function: Detailed description of class functionality
"""
'''
            content = re.sub(fr'class\s+{class_name}(?:\(|:)', class_comment + f'class {class_name}(:', content)
    
    # 添加函数注释模板 | Add function comment template
    func_pattern = r'def\s+(\w+)\s*\('
    functions = re.findall(func_pattern, content)
    
    for func_name in functions:
        if func_name not in ['__init__', '__str__', '__repr__']:
            func_comment_pattern = fr'def\s+{func_name}\s*\(.*?(?:"""|\'\'\')'
            if not re.search(func_comment_pattern, content, re.DOTALL):
                func_comment = f'''
"""
{func_name}函数 - 中文函数描述
{func_name} Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
'''
                content = re.sub(fr'def\s+{func_name}\s*\(', func_comment + f'def {func_name}(', content)
    
    # 保存修改后的文件 | Save modified file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已为 {file_path} 添加中英文注释 | Added bilingual comments to {file_path}")

def main():
    parser = argparse.ArgumentParser(description="自动添加中英文注释 | Automatically add bilingual comments")
    parser.add_argument("path", help="要处理的目录或文件路径 | Path to directory or file to process")
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        if path.suffix == '.py':
            add_bilingual_comments(path)
    else:
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    add_bilingual_comments(file_path)

if __name__ == "__main__":
    main()
