import os
import re

def search_creative_solver_references(root_dir):
    pattern = r'CreativeProblemSolver'
    matches = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if re.search(pattern, content):
                            lines = content.split('\n')
                            for line_num, line in enumerate(lines, 1):
                                if re.search(pattern, line):
                                    matches.append((file_path, line_num, line.strip()))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return matches

if __name__ == "__main__":
    root_dir = "E:\119\Self-Soul-main"
    matches = search_creative_solver_references(root_dir)
    
    if matches:
        print("Found references to CreativeProblemSolver in:")
        for file_path, line_num, line in matches:
            print(f"  - {file_path}:{line_num}: {line}")
    else:
        print("No references to CreativeProblemSolver found.")
