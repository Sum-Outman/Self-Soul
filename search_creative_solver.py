import os
import re

def search_creative_solver_imports(root_dir):
    pattern = r'from\s+core\.creative_problem_solver\s+import\s+CreativeProblemSolver'
    matches = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if re.search(pattern, content):
                            matches.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return matches

if __name__ == "__main__":
    root_dir = "E:\119\Self-Soul-main"
    matches = search_creative_solver_imports(root_dir)
    
    if matches:
        print("Found imports of CreativeProblemSolver in:")
        for match in matches:
            print(f"  - {match}")
    else:
        print("No imports of CreativeProblemSolver found.")
