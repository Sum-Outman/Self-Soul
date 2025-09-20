""" 
Bilingual Comments Checker Tool

Purpose: Check if code files contain bilingual Chinese-English comments
""" 

import os
import re
import argparse
from pathlib import Path

def check_file_comments(file_path):
    """Check bilingual comments in single file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check class comments
    class_pattern = r'class\s+(\w+).*?("""|\'\'\')(.*?)("""|\'\'\')'
    class_matches = re.findall(class_pattern, content, re.DOTALL)
    
    # Check function comments
    func_pattern = r'def\s+(\w+).*?("""|\'\'\')(.*?)("""|\'\'\')'
    func_matches = re.findall(func_pattern, content, re.DOTALL)
    
    issues = []
    
    for match in class_matches + func_matches:
        comment = match[2]
        # Check if contains both Chinese and English
        has_chinese = re.search(r'[\u4e00-\u9fff]', comment)
        has_english = re.search(r'[a-zA-Z]', comment)
        
        if not (has_chinese and has_english):
            if 'class' in str(match):
                issue_type = "Class comment"
            else:
                issue_type = "Function comment"
            
            issues.append(f"{issue_type}: {match[0]} - Missing bilingual comments")
    
    return issues

def main():
    parser = argparse.ArgumentParser(description="Check bilingual comments in code files")
    parser.add_argument("path", help="Path to directory or file to check")
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
        print("Found comment issues:")
        for issue in all_issues:
            print(f"  - {issue}")
        return 1
    else:
        print("All comments check passed")
        return 0

if __name__ == "__main__":
    exit(main())
