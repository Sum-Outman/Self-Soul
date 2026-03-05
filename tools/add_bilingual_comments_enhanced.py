"""
Enhanced Bilingual Comments Tool with Intelligent Analysis

Function: Automatically add intelligent bilingual comments to Python files
基于AST分析和代码语义理解的智能双语注释工具
"""

import os
import re
import ast
import inspect
import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import json


class CodeAnalyzer:
    """Advanced code analyzer using AST for Python syntax understanding"""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.splitlines()
        self.tree = ast.parse(source_code)
        
    def extract_functions(self) -> List[Dict[str, Any]]:
        """Extract all functions from the code with detailed information"""
        functions = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_function(node)
                functions.append(func_info)
        
        return functions
    
    def extract_classes(self) -> List[Dict[str, Any]]:
        """Extract all classes from the code with detailed information"""
        classes = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._analyze_class(node)
                classes.append(class_info)
        
        return classes
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a single function definition"""
        # Get function signature
        func_name = node.name
        
        # Get decorators
        decorators = []
        for decorator in node.decorator_list:
            decorator_str = ast.unparse(decorator) if hasattr(ast, 'unparse') else self._node_to_string(decorator)
            decorators.append(decorator_str)
        
        # Analyze arguments
        args_info = self._analyze_arguments(node.args)
        
        # Analyze return type annotation
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else self._node_to_string(node.returns)
        
        # Get function body (first few lines for context)
        body_preview = self._get_function_body_preview(node)
        
        # Get docstring if exists
        docstring = ast.get_docstring(node)
        
        # Get line numbers
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        return {
            'name': func_name,
            'decorators': decorators,
            'args': args_info,
            'return_type': return_type,
            'body_preview': body_preview,
            'docstring': docstring,
            'start_line': start_line,
            'end_line': end_line,
            'is_async': isinstance(node, ast.AsyncFunctionDef)
        }
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a single class definition"""
        class_name = node.name
        
        # Get base classes
        bases = []
        for base in node.bases:
            base_str = ast.unparse(base) if hasattr(ast, 'unparse') else self._node_to_string(base)
            bases.append(base_str)
        
        # Get decorators
        decorators = []
        for decorator in node.decorator_list:
            decorator_str = ast.unparse(decorator) if hasattr(ast, 'unparse') else self._node_to_string(decorator)
            decorators.append(decorator_str)
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item)
                methods.append(method_info)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Get line numbers
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        return {
            'name': class_name,
            'bases': bases,
            'decorators': decorators,
            'methods': methods,
            'docstring': docstring,
            'start_line': start_line,
            'end_line': end_line
        }
    
    def _analyze_arguments(self, args_node: ast.arguments) -> List[Dict[str, Any]]:
        """Analyze function arguments with type annotations"""
        args_info = []
        
        # Positional arguments
        for arg in args_node.args:
            arg_info = self._analyze_single_argument(arg)
            args_info.append(arg_info)
        
        # Default arguments
        defaults = args_node.defaults
        default_offset = len(args_node.args) - len(defaults)
        
        for i, default in enumerate(defaults):
            idx = default_offset + i
            if idx < len(args_info):
                default_value = ast.unparse(default) if hasattr(ast, 'unparse') else self._node_to_string(default)
                args_info[idx]['default'] = default_value
        
        # *args
        if args_node.vararg:
            arg_info = self._analyze_single_argument(args_node.vararg)
            arg_info['kind'] = 'varargs'
            args_info.append(arg_info)
        
        # **kwargs
        if args_node.kwarg:
            arg_info = self._analyze_single_argument(args_node.kwarg)
            arg_info['kind'] = 'kwargs'
            args_info.append(arg_info)
        
        return args_info
    
    def _analyze_single_argument(self, arg_node) -> Dict[str, Any]:
        """Analyze a single argument"""
        arg_name = arg_node.arg
        
        # Type annotation
        arg_type = None
        if hasattr(arg_node, 'annotation') and arg_node.annotation:
            arg_type = ast.unparse(arg_node.annotation) if hasattr(ast, 'unparse') else self._node_to_string(arg_node.annotation)
        
        return {
            'name': arg_name,
            'type': arg_type,
            'default': None,
            'kind': 'positional'
        }
    
    def _get_function_body_preview(self, node: ast.FunctionDef, max_lines: int = 3) -> str:
        """Get a preview of the function body for context analysis"""
        lines = []
        start_line = node.lineno
        
        # Skip the function definition line and docstring
        body_start = 0
        for i, item in enumerate(node.body):
            if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
                # Skip docstring
                continue
            body_start = i
            break
        
        # Get first few non-docstring lines
        for i in range(body_start, min(len(node.body), body_start + max_lines)):
            item = node.body[i]
            line_no = item.lineno if hasattr(item, 'lineno') else start_line + i + 1
            if line_no - 1 < len(self.lines):
                lines.append(self.lines[line_no - 1])
        
        return '\n'.join(lines)
    
    def _node_to_string(self, node) -> str:
        """Convert AST node to readable string representation"""
        if node is None:
            return ""
        
        # Try ast.unparse first (Python 3.9+)
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
        except:
            pass
        
        # Handle common AST node types
        node_type = type(node).__name__
        
        if node_type == 'Name':
            return node.id
        elif node_type == 'Attribute':
            # Handle attribute access like module.Class
            value_str = self._node_to_string(node.value)
            return f"{value_str}.{node.attr}"
        elif node_type == 'Subscript':
            # Handle subscript like List[str]
            value_str = self._node_to_string(node.value)
            slice_str = self._node_to_string(node.slice)
            return f"{value_str}[{slice_str}]"
        elif node_type == 'Constant':
            return repr(node.value)
        elif node_type == 'Str':
            return repr(node.s)
        elif node_type == 'Num':
            return repr(node.n)
        elif node_type == 'List':
            elements = [self._node_to_string(e) for e in node.elts]
            return f"[{', '.join(elements)}]"
        elif node_type == 'Tuple':
            elements = [self._node_to_string(e) for e in node.elts]
            return f"({', '.join(elements)})"
        elif node_type == 'Dict':
            keys = [self._node_to_string(k) for k in node.keys]
            values = [self._node_to_string(v) for v in node.values]
            pairs = [f"{k}: {v}" for k, v in zip(keys, values)]
            return f"{{{', '.join(pairs)}}}"
        elif node_type == 'Index':
            return self._node_to_string(node.value)
        elif node_type == 'Load':
            return ""
        elif node_type == 'Store':
            return ""
        elif node_type == 'Expr':
            return self._node_to_string(node.value)
        
        # Fallback to ast.dump for unknown types
        try:
            return ast.dump(node)
        except:
            return str(node)


class CommentGenerator:
    """Generate intelligent comments based on code analysis"""
    
    def __init__(self, analyzer: CodeAnalyzer):
        self.analyzer = analyzer
        self.keyword_map = self._build_keyword_map()
    
    def _build_keyword_map(self) -> Dict[str, Dict[str, str]]:
        """Build mapping of keywords to descriptions in both languages"""
        return {
            'get': {
                'en': 'Retrieve',
                'zh': '获取'
            },
            'set': {
                'en': 'Set or update',
                'zh': '设置或更新'
            },
            'create': {
                'en': 'Create new',
                'zh': '创建新的'
            },
            'update': {
                'en': 'Update existing',
                'zh': '更新现有的'
            },
            'delete': {
                'en': 'Delete or remove',
                'zh': '删除或移除'
            },
            'validate': {
                'en': 'Validate or check',
                'zh': '验证或检查'
            },
            'calculate': {
                'en': 'Calculate or compute',
                'zh': '计算'
            },
            'process': {
                'en': 'Process or handle',
                'zh': '处理'
            },
            'analyze': {
                'en': 'Analyze or examine',
                'zh': '分析或检查'
            },
            'generate': {
                'en': 'Generate or produce',
                'zh': '生成或产生'
            },
            'initialize': {
                'en': 'Initialize or set up',
                'zh': '初始化或设置'
            },
            'load': {
                'en': 'Load or read',
                'zh': '加载或读取'
            },
            'save': {
                'en': 'Save or write',
                'zh': '保存或写入'
            },
            'find': {
                'en': 'Find or locate',
                'zh': '查找或定位'
            },
            'search': {
                'en': 'Search or look for',
                'zh': '搜索或查找'
            },
            'parse': {
                'en': 'Parse or interpret',
                'zh': '解析或解释'
            },
            'format': {
                'en': 'Format or arrange',
                'zh': '格式化或排列'
            },
            'convert': {
                'en': 'Convert or transform',
                'zh': '转换或变换'
            },
            'merge': {
                'en': 'Merge or combine',
                'zh': '合并或组合'
            },
            'split': {
                'en': 'Split or divide',
                'zh': '分割或分开'
            },
            'filter': {
                'en': 'Filter or select',
                'zh': '过滤或选择'
            },
            'sort': {
                'en': 'Sort or order',
                'zh': '排序或整理'
            },
            'compare': {
                'en': 'Compare or contrast',
                'zh': '比较或对比'
            },
            'check': {
                'en': 'Check or verify',
                'zh': '检查或验证'
            },
            'handle': {
                'en': 'Handle or manage',
                'zh': '处理或管理'
            },
            'execute': {
                'en': 'Execute or run',
                'zh': '执行或运行'
            },
            'evaluate': {
                'en': 'Evaluate or assess',
                'zh': '评估或评价'
            },
            'optimize': {
                'en': 'Optimize or improve',
                'zh': '优化或改进'
            },
            'monitor': {
                'en': 'Monitor or track',
                'zh': '监控或跟踪'
            },
            'log': {
                'en': 'Log or record',
                'zh': '记录或日志'
            },
            'report': {
                'en': 'Report or summarize',
                'zh': '报告或总结'
            }
        }
    
    def generate_function_comment(self, func_info: Dict[str, Any]) -> str:
        """Generate intelligent bilingual comment for a function"""
        func_name = func_info['name']
        
        # Skip special methods
        if func_name.startswith('__') and func_name.endswith('__'):
            return ""
        
        # Generate function description based on name and context
        description_en, description_zh = self._infer_function_description(func_info)
        
        # Build args section
        args_section = self._generate_args_section(func_info['args'])
        
        # Build returns section
        returns_section = self._generate_returns_section(func_info)
        
        # Build decorators section if any
        decorators_section = ""
        if func_info['decorators']:
            decorators_list = '\n'.join([f"    @{deco}" for deco in func_info['decorators']])
            decorators_section = f"\nDecorators:\n{decorators_list}"
        
        comment = f'''"""
{func_name} Function - {description_en}

功能：{description_zh}

Args:
{args_section}
Returns:
{returns_section}{decorators_section}
"""
'''
        return comment
    
    def generate_class_comment(self, class_info: Dict[str, Any]) -> str:
        """Generate intelligent bilingual comment for a class"""
        class_name = class_info['name']
        
        # Generate class description
        description_en, description_zh = self._infer_class_description(class_info)
        
        # Build base classes section
        bases_section = ""
        if class_info['bases']:
            bases_str = ', '.join(class_info['bases'])
            bases_section = f"\nInherits from: {bases_str}"
        
        # Build decorators section
        decorators_section = ""
        if class_info['decorators']:
            decorators_list = '\n'.join([f"    @{deco}" for deco in class_info['decorators']])
            decorators_section = f"\nDecorators:\n{decorators_list}"
        
        comment = f'''"""
{class_name} Class - {description_en}

类功能：{description_zh}{bases_section}{decorators_section}

Methods:
{self._generate_methods_summary(class_info['methods'])}
"""
'''
        return comment
    
    def _infer_function_description(self, func_info: Dict[str, Any]) -> Tuple[str, str]:
        """Infer function description from name, arguments and body"""
        func_name = func_info['name'].lower()
        
        # Check for common patterns in function name
        for keyword, descriptions in self.keyword_map.items():
            if keyword in func_name:
                return descriptions['en'], descriptions['zh']
        
        # Analyze based on function name patterns
        if func_name.startswith('is_') or func_name.startswith('has_'):
            return "Check condition", "检查条件"
        elif func_name.startswith('get_'):
            return "Get value", "获取值"
        elif func_name.startswith('set_'):
            return "Set value", "设置值"
        elif func_name.startswith('create_'):
            return "Create new instance", "创建新实例"
        elif func_name.startswith('update_'):
            return "Update existing", "更新现有内容"
        elif func_name.startswith('delete_') or func_name.startswith('remove_'):
            return "Delete item", "删除项目"
        elif func_name.startswith('find_') or func_name.startswith('search_'):
            return "Find or locate", "查找或定位"
        elif func_name.startswith('parse_'):
            return "Parse or interpret", "解析或解释"
        elif func_name.startswith('format_'):
            return "Format or arrange", "格式化或排列"
        elif func_name.startswith('convert_') or func_name.startswith('transform_'):
            return "Convert or transform", "转换或变换"
        elif func_name.startswith('validate_') or func_name.startswith('check_'):
            return "Validate or check", "验证或检查"
        elif func_name.startswith('calculate_') or func_name.startswith('compute_'):
            return "Calculate or compute", "计算"
        elif func_name.startswith('process_'):
            return "Process or handle", "处理"
        elif func_name.startswith('analyze_'):
            return "Analyze or examine", "分析或检查"
        elif func_name.startswith('generate_'):
            return "Generate or produce", "生成或产生"
        elif func_name.startswith('initialize_') or func_name.startswith('init_'):
            return "Initialize or set up", "初始化或设置"
        elif func_name.startswith('load_'):
            return "Load or read", "加载或读取"
        elif func_name.startswith('save_'):
            return "Save or write", "保存或写入"
        elif func_name.startswith('merge_'):
            return "Merge or combine", "合并或组合"
        elif func_name.startswith('split_'):
            return "Split or divide", "分割或分开"
        elif func_name.startswith('filter_'):
            return "Filter or select", "过滤或选择"
        elif func_name.startswith('sort_'):
            return "Sort or order", "排序或整理"
        elif func_name.startswith('compare_'):
            return "Compare or contrast", "比较或对比"
        elif func_name.startswith('execute_') or func_name.startswith('run_'):
            return "Execute or run", "执行或运行"
        elif func_name.startswith('evaluate_'):
            return "Evaluate or assess", "评估或评价"
        elif func_name.startswith('optimize_'):
            return "Optimize or improve", "优化或改进"
        elif func_name.startswith('monitor_'):
            return "Monitor or track", "监控或跟踪"
        elif func_name.startswith('log_'):
            return "Log or record", "记录或日志"
        elif func_name.startswith('report_'):
            return "Report or summarize", "报告或总结"
        
        # Analyze based on return type
        return_type = func_info.get('return_type', '')
        if return_type:
            if return_type.lower() in ['bool', 'boolean']:
                return "Check condition and return boolean", "检查条件并返回布尔值"
            elif return_type.lower() in ['int', 'float', 'number']:
                return "Calculate and return numeric value", "计算并返回数值"
            elif return_type.lower() in ['str', 'string']:
                return "Process and return string", "处理并返回字符串"
            elif return_type.lower() in ['list', 'array']:
                return "Process and return list", "处理并返回列表"
            elif return_type.lower() in ['dict', 'dictionary']:
                return "Process and return dictionary", "处理并返回字典"
        
        # Analyze based on arguments
        args = func_info.get('args', [])
        if len(args) == 0:
            return "Perform operation without parameters", "执行无参数操作"
        
        # Check body for clues
        body_preview = func_info.get('body_preview', '').lower()
        if body_preview:
            if 'return' in body_preview:
                if 'calculate' in body_preview or 'compute' in body_preview:
                    return "Calculate value", "计算值"
                elif 'validate' in body_preview or 'check' in body_preview:
                    return "Validate and return result", "验证并返回结果"
                return "Process and return value", "处理并返回值"
        
        # Default descriptions
        return "Perform operation", "执行操作"
    
    def _infer_class_description(self, class_info: Dict[str, Any]) -> Tuple[str, str]:
        """Infer class description from name and methods"""
        class_name = class_info['name'].lower()
        
        # Common class patterns based on name suffixes
        if class_name.endswith('manager'):
            return "Manager class for handling operations", "处理操作的管理器类"
        elif class_name.endswith('service'):
            return "Service class providing functionality", "提供功能的服务类"
        elif class_name.endswith('model'):
            return "Data model class", "数据模型类"
        elif class_name.endswith('controller'):
            return "Controller class for managing flow", "管理流程的控制器类"
        elif class_name.endswith('handler'):
            return "Handler class for processing requests", "处理请求的处理器类"
        elif class_name.endswith('factory'):
            return "Factory class for creating instances", "创建实例的工厂类"
        elif class_name.endswith('utils') or class_name.endswith('helper'):
            return "Utility class with helper functions", "包含辅助函数的工具类"
        elif class_name.endswith('processor'):
            return "Data processor class", "数据处理器类"
        elif class_name.endswith('analyzer'):
            return "Data analyzer class", "数据分析器类"
        elif class_name.endswith('generator'):
            return "Content generator class", "内容生成器类"
        elif class_name.endswith('validator'):
            return "Data validator class", "数据验证器类"
        elif class_name.endswith('parser'):
            return "Data parser class", "数据解析器类"
        elif class_name.endswith('formatter'):
            return "Data formatter class", "数据格式化器类"
        elif class_name.endswith('converter'):
            return "Data converter class", "数据转换器类"
        elif class_name.endswith('calculator'):
            return "Calculator class", "计算器类"
        elif class_name.endswith('client'):
            return "Client class for external services", "外部服务客户端类"
        elif class_name.endswith('adapter'):
            return "Adapter class for compatibility", "兼容性适配器类"
        elif class_name.endswith('wrapper'):
            return "Wrapper class for encapsulation", "封装包装器类"
        elif class_name.endswith('interface'):
            return "Interface definition class", "接口定义类"
        elif class_name.endswith('abstract'):
            return "Abstract base class", "抽象基类"
        elif class_name.endswith('base'):
            return "Base class for inheritance", "继承用的基类"
        elif class_name.endswith('mixin'):
            return "Mixin class for adding functionality", "添加功能的混入类"
        elif class_name.endswith('decorator'):
            return "Decorator class", "装饰器类"
        elif class_name.endswith('exception'):
            return "Exception class", "异常类"
        elif class_name.endswith('error'):
            return "Error handling class", "错误处理类"
        
        # Analyze methods
        method_names = [method['name'].lower() for method in class_info['methods']]
        
        # Check for common method patterns
        if any('get' in name or 'fetch' in name or 'read' in name for name in method_names):
            if any('set' in name or 'update' in name or 'write' in name for name in method_names):
                return "Data access and manipulation class", "数据访问和操作类"
            return "Data retrieval class", "数据检索类"
        
        if any('create' in name or 'new' in name for name in method_names):
            return "Object creation class", "对象创建类"
        
        if any('process' in name for name in method_names):
            return "Data processing class", "数据处理类"
        
        if any('validate' in name or 'check' in name for name in method_names):
            return "Validation class", "验证类"
        
        if any('calculate' in name or 'compute' in name for name in method_names):
            return "Calculation class", "计算类"
        
        if any('analyze' in name for name in method_names):
            return "Analysis class", "分析类"
        
        # Check base classes
        bases = class_info.get('bases', [])
        if bases:
            base_names = ' '.join(bases).lower()
            if 'exception' in base_names or 'error' in base_names:
                return "Exception or error class", "异常或错误类"
            elif 'abc' in base_names or 'abstract' in base_names:
                return "Abstract base class", "抽象基类"
        
        return "Implementation class", "实现类"
    
    def _generate_args_section(self, args_info: List[Dict[str, Any]]) -> str:
        """Generate formatted Args section"""
        if not args_info:
            return "    None"
        
        args_lines = []
        for arg in args_info:
            arg_line = f"    {arg['name']}"
            
            # Add type annotation if available
            if arg['type']:
                arg_line += f" ({arg['type']})"
            
            # Add default value if available
            if arg['default']:
                arg_line += f": Default is {arg['default']}"
            else:
                arg_line += ": Parameter description"
            
            args_lines.append(arg_line)
        
        return '\n'.join(args_lines)
    
    def _generate_returns_section(self, func_info: Dict[str, Any]) -> str:
        """Generate formatted Returns section"""
        return_type = func_info['return_type']
        
        if return_type:
            return f"    {return_type}: Return value description"
        else:
            return "    None or result: Return value description"
    
    def _generate_methods_summary(self, methods: List[Dict[str, Any]]) -> str:
        """Generate summary of class methods"""
        if not methods:
            return "    No public methods"
        
        method_lines = []
        for method in methods:
            method_name = method['name']
            if not (method_name.startswith('__') and method_name.endswith('__')):
                method_lines.append(f"    {method_name}(): Method description")
        
        return '\n'.join(method_lines) if method_lines else "    No public methods"


class BilingualCommentProcessor:
    """Main processor for adding bilingual comments to Python files"""
    
    def __init__(self):
        self.analyzer = None
        self.generator = None
    
    def process_file(self, file_path: str) -> bool:
        """Process a single Python file with intelligent comment insertion"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip if already has Apache license header (but still process for other comments)
            has_license = 'Licensed under the Apache License' in content
            
            # Analyze code
            self.analyzer = CodeAnalyzer(content)
            self.generator = CommentGenerator(self.analyzer)
            
            # Extract classes and functions
            classes = self.analyzer.extract_classes()
            functions = self.analyzer.extract_functions()
            
            # Generate comments and insert them
            modified_content = self._add_comments_to_content(content, classes, functions, has_license)
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            print(f"Added intelligent bilingual comments to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _add_comments_to_content(self, content: str, classes: List[Dict[str, Any]], 
                                functions: List[Dict[str, Any]], has_license: bool) -> str:
        """Add generated comments to the content at appropriate locations"""
        lines = content.splitlines(keepends=True)
        
        # Prepare insertion points (line number -> comment)
        insertions = {}
        
        # Collect all entities that need comments
        entities = []
        
        # Add classes
        for class_info in classes:
            # Skip if already has docstring
            if class_info['docstring']:
                continue
            
            class_comment = self.generator.generate_class_comment(class_info)
            if class_comment:
                # AST line numbers are 1-indexed, insert before the class definition
                insert_line = class_info['start_line'] - 1  # Convert to 0-indexed
                entities.append((insert_line, class_comment, 'class', class_info['name']))
        
        # Add functions (only top-level functions, not methods)
        for func_info in functions:
            # Skip if already has docstring or is a method (handled by class)
            if func_info['docstring']:
                continue
            
            # Check if this function is inside a class (by checking line ranges)
            is_method = False
            for class_info in classes:
                if (func_info['start_line'] >= class_info['start_line'] and 
                    func_info['end_line'] <= class_info['end_line']):
                    is_method = True
                    break
            
            if not is_method:
                func_comment = self.generator.generate_function_comment(func_info)
                if func_comment:
                    insert_line = func_info['start_line'] - 1  # Convert to 0-indexed
                    entities.append((insert_line, func_comment, 'function', func_info['name']))
        
        # Sort by line number in descending order (insert from bottom to top to preserve line numbers)
        entities.sort(key=lambda x: x[0], reverse=True)
        
        # Apply insertions
        for insert_line, comment, entity_type, entity_name in entities:
            # Check if there's already a comment at this location
            line_before = lines[insert_line].strip() if insert_line < len(lines) else ""
            
            # Skip if there's already a docstring or comment
            if '"""' in line_before or "'''" in line_before:
                print(f"  Skipping {entity_type} {entity_name} at line {insert_line+1} (already has docstring)")
                continue
            
            # Insert the comment
            lines.insert(insert_line, comment + '\n')
            print(f"  Added comment for {entity_type} {entity_name} at line {insert_line+1}")
        
        # Add license header if not present
        if not has_license:
            header = self._generate_license_header()
            lines.insert(0, header + '\n\n')
            print(f"  Added license header")
        
        return ''.join(lines)
    
    def _generate_license_header(self) -> str:
        """Generate Apache license header"""
        return '''"""
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
智能双语注释工具增强版 - 基于AST分析和代码语义理解
Enhanced Bilingual Comments Tool - Based on AST analysis and code semantic understanding

提供智能注释生成功能，基于函数签名、类型注解和代码上下文
Provides intelligent comment generation based on function signatures, type annotations, and code context
"""'''


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced bilingual comments tool with intelligent analysis",
        epilog="Example: python add_bilingual_comments_enhanced.py core/"
    )
    parser.add_argument(
        "path",
        help="Path to directory or Python file to process"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - analyze but don't modify files"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    processor = BilingualCommentProcessor()
    
    if path.is_file() and path.suffix == '.py':
        if args.test:
            print(f"Test mode: Would process {path}")
            return
        processor.process_file(str(path))
    elif path.is_dir():
        for py_file in path.rglob("*.py"):
            if args.test:
                print(f"Test mode: Would process {py_file}")
            else:
                processor.process_file(str(py_file))
    else:
        print(f"Error: {path} is not a Python file or directory")


if __name__ == "__main__":
    main()
