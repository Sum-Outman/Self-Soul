"""
中英文注释检查工具
Bilingual Comments Checker Tool

功能：检查代码文件是否包含中英文对照注释
Function: Check if code files contain bilingual comments
"""

import os
import re
import argparse
from pathlib import Path

def check_file_comments(file_path):
    """检查单个文件的中英文注释 | Check bilingual comments in single file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查类注释 | Check class comments
    class_pattern = r'class\s+(\w+).*?("""|\'\'\')(.*?)("""|\'\'\')'
    class_matches = re.findall(class_pattern, content, re.DOTALL)
    
    # 检查函数注释 | Check function comments
    func_pattern = r'def\s+(\w+).*?("""|\'\'\')(.*?)("""|\'\'\')'
    func_matches = re.findall(func_pattern, content, re.DOTALL)
    
    issues = []
    
    for match in class_matches + func_matches:
        comment = match[2]
        # 检查是否包含中英文 | Check if contains both Chinese and English
        has_chinese = re.search(r'[\u4e00-\u9fff]', comment)
        has_english = re.search(r'[a-zA-Z]', comment)
        
        if not (has_chinese and has_english):
            if 'class' in str(match):
                issue_type = "类注释 | Class comment"
            else:
                issue_type = "函数注释 | Function comment"
            
            issues.append(f"{issue_type}: {match[0]} - 缺少中英文对照 | Missing bilingual comments")
    
    return issues

def main():
    parser = argparse.ArgumentParser(description="检查代码文件的中英文注释 | Check bilingual comments in code files")
    parser.add_argument("path", help="要检查的目录或文件路径 | Path to directory or file to check")
    args = parser.parse_args()
    
    path = Path(args.path)
    all_issues = []
    
    if path.is_file():
        if path.suffix == '.py':
            issues = check_file_comments(path)
            all_issues.extend(issues)
    else:
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    issues = check_file_comments(file_path)
                    all_issues.extend(issues)
    
    if all_issues:
        print("发现注释问题 | Found comment issues:")
        for issue in all_issues:
            print(f"  - {issue}")
        return 1
    else:
        print("所有注释检查通过 | All comments check passed")
        return 0

if __name__ == "__main__":
    exit(main())
