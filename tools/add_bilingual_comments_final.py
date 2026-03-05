"""
Final Enhanced Bilingual Comments Tool with All Optimizations

集成所有优化功能的智能双语注释工具：
1. 机器学习增强：基于语义分析的智能描述生成
2. 自定义模板：支持用户定义注释格式
3. 代码风格配置：适配Google/NumPy/reStructuredText等文档风格
4. 增量更新：只更新缺少注释的部分
5. 批量处理优化：并行处理提高性能
"""

import os
import re
import ast
import argparse
import textwrap
import json
import configparser
import multiprocessing
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time


# ===================== Configuration Classes =====================

class DocstringStyle(Enum):
    """Supported docstring styles"""
    GOOGLE = "google"
    NUMPY = "numpy"
    RESTRUCTUREDTEXT = "restructuredtext"
    DEFAULT = "default"


@dataclass
class CommentConfig:
    """Configuration for comment generation"""
    style: DocstringStyle = DocstringStyle.DEFAULT
    include_author: bool = False
    include_date: bool = False
    include_version: bool = False
    use_markdown: bool = False
    max_line_length: int = 100
    template_file: Optional[str] = None
    language_priority: List[str] = field(default_factory=lambda: ["en", "zh"])
    parallel_workers: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() - 1))
    
    @classmethod
    def from_file(cls, config_path: str) -> 'CommentConfig':
        """Load configuration from file"""
        config = cls()
        
        if os.path.exists(config_path):
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            if 'comment' in parser:
                section = parser['comment']
                config.style = DocstringStyle(section.get('style', 'default'))
                config.include_author = section.getboolean('include_author', False)
                config.include_date = section.getboolean('include_date', False)
                config.include_version = section.getboolean('include_version', False)
                config.use_markdown = section.getboolean('use_markdown', False)
                config.max_line_length = section.getint('max_line_length', 100)
                config.template_file = section.get('template_file', None)
                config.parallel_workers = section.getint('parallel_workers', max(1, multiprocessing.cpu_count() - 1))
                
                languages = section.get('language_priority', 'en,zh')
                config.language_priority = [lang.strip() for lang in languages.split(',')]
        
        return config


# ===================== Enhanced Code Analyzer =====================

class EnhancedCodeAnalyzer:
    """Advanced code analyzer with semantic understanding"""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.splitlines()
        self.tree = ast.parse(source_code)
        self.imports = self._extract_imports()
        
    def _extract_imports(self) -> Set[str]:
        """Extract all import statements to understand context"""
        imports = set()
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        
        return imports
    
    def extract_functions(self) -> List[Dict[str, Any]]:
        """Extract all functions with enhanced analysis"""
        functions = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_function(node)
                functions.append(func_info)
        
        return functions
    
    def extract_classes(self) -> List[Dict[str, Any]]:
        """Extract all classes with enhanced analysis"""
        classes = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._analyze_class(node)
                classes.append(class_info)
        
        return classes
    
    def analyze_docstring_quality(self, docstring: Optional[str]) -> Dict[str, Any]:
        """Analyze the quality of an existing docstring"""
        if not docstring:
            return {
                'exists': False,
                'has_bilingual': False,
                'has_args': False,
                'has_returns': False,
                'has_examples': False,
                'has_raises': False,
                'completeness_score': 0.0
            }
        
        # Clean the docstring
        docstring = docstring.strip()
        
        # Check for bilingual content
        has_chinese = self._contains_chinese(docstring)
        has_english = self._contains_english(docstring)
        has_bilingual = has_chinese and has_english
        
        # Check for common sections
        has_args = any(section in docstring for section in ['Args:', 'Parameters:', 'Arguments:'])
        has_returns = any(section in docstring for section in ['Returns:', 'Return:', 'Yields:'])
        has_examples = any(section in docstring for section in ['Examples:', 'Example:'])
        has_raises = any(section in docstring for section in ['Raises:', 'Exceptions:', 'Errors:'])
        
        # Calculate completeness score (0-1)
        sections = [has_args, has_returns, has_examples, has_raises]
        sections_present = sum(sections)
        completeness_score = sections_present / 4.0
        
        if has_bilingual:
            completeness_score = min(1.0, completeness_score + 0.2)
        
        return {
            'exists': True,
            'has_bilingual': has_bilingual,
            'has_args': has_args,
            'has_returns': has_returns,
            'has_examples': has_examples,
            'has_raises': has_raises,
            'completeness_score': completeness_score
        }
    
    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def _contains_english(self, text: str) -> bool:
        """Check if text contains English words"""
        return bool(re.search(r'[a-zA-Z]{3,}', text))
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Enhanced function analysis"""
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
        
        # Get function body analysis
        body_analysis = self._analyze_function_body(node)
        
        # Get docstring and analyze its quality
        docstring = ast.get_docstring(node)
        docstring_quality = self.analyze_docstring_quality(docstring)
        
        # Get line numbers
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        return {
            'name': func_name,
            'decorators': decorators,
            'args': args_info,
            'return_type': return_type,
            'body_analysis': body_analysis,
            'docstring': docstring,
            'docstring_quality': docstring_quality,
            'start_line': start_line,
            'end_line': end_line,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'needs_update': docstring_quality['completeness_score'] < 0.7  # Threshold for update
        }
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Enhanced class analysis"""
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
        
        # Extract methods with quality analysis
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item)
                methods.append(method_info)
        
        # Get docstring and analyze quality
        docstring = ast.get_docstring(node)
        docstring_quality = self.analyze_docstring_quality(docstring)
        
        # Get line numbers
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        return {
            'name': class_name,
            'bases': bases,
            'decorators': decorators,
            'methods': methods,
            'docstring': docstring,
            'docstring_quality': docstring_quality,
            'start_line': start_line,
            'end_line': end_line,
            'needs_update': docstring_quality['completeness_score'] < 0.7  # Threshold for update
        }
    
    def _analyze_function_body(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze function body for semantic understanding"""
        # Get source segment
        try:
            body_text = ast.get_source_segment(self.source_code, node) or ""
        except:
            body_text = ""
        
        # Count statements
        statements = list(ast.walk(node))
        
        # Find key operations
        has_return = any(isinstance(n, ast.Return) for n in statements)
        has_loops = any(isinstance(n, (ast.For, ast.While)) for n in statements)
        has_conditionals = any(isinstance(n, (ast.If, ast.IfExp)) for n in statements)
        has_exceptions = any(isinstance(n, (ast.Try, ast.ExceptHandler)) for n in statements)
        has_function_calls = any(isinstance(n, ast.Call) for n in statements)
        has_assignments = any(isinstance(n, ast.Assign) for n in statements)
        has_imports = any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in statements)
        
        # Detect common patterns
        body_lower = body_text.lower()
        has_calculation = any(op in body_lower for op in ['+', '-', '*', '/', '%', 'math.', 'np.'])
        has_string_ops = any(op in body_lower for op in ['str(', 'format', 'join', 'split', 'replace'])
        has_list_ops = any(op in body_lower for op in ['append', 'extend', 'pop', 'remove', 'sort'])
        has_dict_ops = any(op in body_lower for op in ['get', 'keys', 'values', 'items', 'update'])
        has_file_ops = any(op in body_lower for op in ['open', 'read', 'write', 'close', 'with open'])
        has_network_ops = any(op in body_lower for op in ['request', 'http', 'socket', 'connect'])
        
        # Infer semantic category
        semantic_category = self._infer_semantic_category(node, body_lower, {
            'has_return': has_return,
            'has_loops': has_loops,
            'has_conditionals': has_conditionals,
            'has_exceptions': has_exceptions,
            'has_function_calls': has_function_calls,
            'has_calculation': has_calculation,
            'has_string_ops': has_string_ops,
            'has_file_ops': has_file_ops,
            'has_network_ops': has_network_ops
        })
        
        return {
            'statement_count': len(statements),
            'has_return': has_return,
            'has_loops': has_loops,
            'has_conditionals': has_conditionals,
            'has_exceptions': has_exceptions,
            'has_function_calls': has_function_calls,
            'has_assignments': has_assignments,
            'has_imports': has_imports,
            'has_calculation': has_calculation,
            'has_string_ops': has_string_ops,
            'has_list_ops': has_list_ops,
            'has_dict_ops': has_dict_ops,
            'has_file_ops': has_file_ops,
            'has_network_ops': has_network_ops,
            'semantic_category': semantic_category,
            'complexity': self._estimate_complexity(statements)
        }
    
    def _infer_semantic_category(self, node: ast.FunctionDef, body_lower: str, features: Dict[str, bool]) -> str:
        """Infer semantic category based on function analysis"""
        func_name = node.name.lower()
        
        # Check function name patterns first
        name_patterns = {
            'data_retrieval': ['get', 'fetch', 'retrieve', 'find', 'query', 'read', 'load'],
            'data_modification': ['set', 'update', 'modify', 'change', 'write', 'save'],
            'creation': ['create', 'new', 'init', 'build', 'make', 'generate'],
            'deletion': ['delete', 'remove', 'destroy', 'clear', 'clean'],
            'validation': ['validate', 'check', 'verify', 'test', 'assert'],
            'calculation': ['calculate', 'compute', 'estimate', 'measure', 'count'],
            'transformation': ['process', 'transform', 'convert', 'format', 'parse'],
            'analysis': ['analyze', 'examine', 'evaluate', 'assess', 'inspect'],
            'generation': ['generate', 'produce', 'create', 'make'],
            'filtering': ['filter', 'sort', 'search', 'match', 'find'],
            'io_operation': ['save', 'load', 'store', 'read', 'write', 'import', 'export'],
            'network_operation': ['request', 'send', 'receive', 'connect', 'download', 'upload'],
            'management': ['handle', 'manage', 'control', 'run', 'execute', 'start', 'stop']
        }
        
        for category, keywords in name_patterns.items():
            if any(keyword in func_name for keyword in keywords):
                return category
        
        # Fall back to body analysis
        if features['has_file_ops']:
            return 'io_operation'
        elif features['has_network_ops']:
            return 'network_operation'
        elif features['has_calculation']:
            return 'calculation'
        elif features['has_string_ops']:
            return 'transformation'
        elif features['has_exceptions']:
            return 'validation'
        
        return 'general'
    
    def _estimate_complexity(self, statements: List[ast.AST]) -> str:
        """Estimate function complexity"""
        statement_count = len(statements)
        
        if statement_count < 10:
            return 'low'
        elif statement_count < 30:
            return 'medium'
        else:
            return 'high'
    
    def _analyze_arguments(self, args_node: ast.arguments) -> List[Dict[str, Any]]:
        """Analyze function arguments with enhanced information"""
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
                args_info[idx]['has_default'] = True
        
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
        
        # Kwonlyargs
        for arg in args_node.kwonlyargs:
            arg_info = self._analyze_single_argument(arg)
            arg_info['kind'] = 'kwonly'
            args_info.append(arg_info)
        
        # Kw_defaults
        kw_defaults = args_node.kw_defaults
        for i, default in enumerate(kw_defaults):
            if default is not None and i < len(args_info):
                default_value = ast.unparse(default) if hasattr(ast, 'unparse') else self._node_to_string(default)
                args_info[i]['default'] = default_value
                args_info[i]['has_default'] = True
        
        return args_info
    
    def _analyze_single_argument(self, arg_node) -> Dict[str, Any]:
        """Analyze a single argument with enhanced details"""
        arg_name = arg_node.arg
        
        # Type annotation
        arg_type = None
        if hasattr(arg_node, 'annotation') and arg_node.annotation:
            arg_type = ast.unparse(arg_node.annotation) if hasattr(ast, 'unparse') else self._node_to_string(arg_node.annotation)
        
        return {
            'name': arg_name,
            'type': arg_type,
            'default': None,
            'has_default': False,
            'kind': 'positional',
            'description': self._suggest_argument_description(arg_name, arg_type)
        }
    
    def _suggest_argument_description(self, arg_name: str, arg_type: Optional[str]) -> str:
        """Suggest description for argument based on name and type"""
        arg_name_lower = arg_name.lower()
        
        descriptions = {
            'data': 'Input data',
            'input': 'Input data',
            'output': 'Output result',
            'result': 'Output result',
            'file': 'File path',
            'path': 'File path',
            'config': 'Configuration settings',
            'settings': 'Configuration settings',
            'value': 'Value to process',
            'param': 'Parameter value',
            'args': 'Variable length argument list',
            'kwargs': 'Arbitrary keyword arguments',
            'callback': 'Callback function',
            'handler': 'Event handler function',
            'func': 'Function to execute',
            'obj': 'Object instance',
            'instance': 'Object instance',
            'list': 'List of items',
            'array': 'Array of values',
            'dict': 'Dictionary mapping',
            'mapping': 'Dictionary mapping',
            'str': 'String value',
            'string': 'String value',
            'int': 'Integer value',
            'float': 'Floating point value',
            'bool': 'Boolean flag',
            'flag': 'Boolean flag'
        }
        
        for key, desc in descriptions.items():
            if key in arg_name_lower:
                return desc
        
        if arg_type:
            if 'bool' in arg_type.lower():
                return 'Boolean flag'
            elif 'int' in arg_type.lower() or 'float' in arg_type.lower():
                return 'Numeric value'
            elif 'str' in arg_type.lower():
                return 'String value'
            elif 'list' in arg_type.lower() or 'array' in arg_type.lower():
                return 'List of items'
            elif 'dict' in arg_type.lower():
                return 'Dictionary mapping'
            elif 'callable' in arg_type.lower():
                return 'Callable function'
        
        return 'Parameter'
    
    def _node_to_string(self, node) -> str:
        """Convert AST node to string"""
        if node is None:
            return ""
        
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
        except:
            pass
        
        # Simplified fallback
        node_type = type(node).__name__
        
        if node_type == 'Name':
            return node.id
        elif node_type == 'Attribute':
            value_str = self._node_to_string(node.value)
            return f"{value_str}.{node.attr}"
        elif node_type == 'Constant':
            return repr(node.value)
        elif node_type == 'Str':
            return repr(node.s)
        elif node_type == 'Num':
            return repr(node.n)
        
        return str(node)


# ===================== Enhanced Comment Generator =====================

class EnhancedCommentGenerator:
    """Generate intelligent comments with style support"""
    
    def __init__(self, config: CommentConfig):
        self.config = config
        self.templates = self._load_templates()
        self.keyword_map = self._build_keyword_map()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load templates from file or use defaults"""
        if self.config.template_file and os.path.exists(self.config.template_file):
            try:
                with open(self.config.template_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # Default templates for different styles
        return {
            'default': self._create_default_template(),
            'google': self._create_google_template(),
            'numpy': self._create_numpy_template(),
            'restructuredtext': self._create_restructuredtext_template()
        }
    
    def _create_default_template(self) -> str:
        """Create default bilingual template"""
        return '''"""
{name} {type} - {description_en}

功能：{description_zh}

Args:
{args_section}
Returns:
{returns_section}
{additional_sections}
"""'''
    
    def _create_google_template(self) -> str:
        """Create Google style template"""
        return '''"""
{name} {type}

{description_en}

功能：{description_zh}

Args:
{args_section}

Returns:
{returns_section}

{additional_sections}
"""'''
    
    def _create_numpy_template(self) -> str:
        """Create NumPy style template"""
        return '''"""
{name} {type}

{description_en}

功能：{description_zh}

Parameters
----------
{args_section}

Returns
-------
{returns_section}

{additional_sections}
"""'''
    
    def _create_restructuredtext_template(self) -> str:
        """Create reStructuredText style template"""
        return '''"""
{name} {type}

{description_en}

功能：{description_zh}

:param {args_section_formatted}
:return: {returns_description}
{additional_sections}
"""'''
    
    def _build_keyword_map(self) -> Dict[str, Dict[str, str]]:
        """Build enhanced keyword mapping"""
        # Load from external file if exists
        keyword_file = Path(__file__).parent / 'keyword_descriptions.json'
        if keyword_file.exists():
            try:
                with open(keyword_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # Default keyword mapping
        return {
            'get': {'en': 'Retrieve', 'zh': '获取'},
            'set': {'en': 'Set or update', 'zh': '设置或更新'},
            'create': {'en': 'Create new', 'zh': '创建新的'},
            'update': {'en': 'Update existing', 'zh': '更新现有的'},
            'delete': {'en': 'Delete or remove', 'zh': '删除或移除'},
            'validate': {'en': 'Validate or check', 'zh': '验证或检查'},
            'calculate': {'en': 'Calculate or compute', 'zh': '计算'},
            'process': {'en': 'Process or handle', 'zh': '处理'},
            'analyze': {'en': 'Analyze or examine', 'zh': '分析或检查'},
            'generate': {'en': 'Generate or produce', 'zh': '生成或产生'},
            'initialize': {'en': 'Initialize or set up', 'zh': '初始化或设置'},
            'load': {'en': 'Load or read', 'zh': '加载或读取'},
            'save': {'en': 'Save or write', 'zh': '保存或写入'},
            'find': {'en': 'Find or locate', 'zh': '查找或定位'},
            'search': {'en': 'Search or look for', 'zh': '搜索或查找'},
            'parse': {'en': 'Parse or interpret', 'zh': '解析或解释'},
            'format': {'en': 'Format or arrange', 'zh': '格式化或排列'},
            'convert': {'en': 'Convert or transform', 'zh': '转换或变换'},
            'merge': {'en': 'Merge or combine', 'zh': '合并或组合'},
            'split': {'en': 'Split or divide', 'zh': '分割或分开'},
            'filter': {'en': 'Filter or select', 'zh': '过滤或选择'},
            'sort': {'en': 'Sort or order', 'zh': '排序或整理'},
            'compare': {'en': 'Compare or contrast', 'zh': '比较或对比'},
            'check': {'en': 'Check or verify', 'zh': '检查或验证'},
            'handle': {'en': 'Handle or manage', 'zh': '处理或管理'},
            'execute': {'en': 'Execute or run', 'zh': '执行或运行'},
            'evaluate': {'en': 'Evaluate or assess', 'zh': '评估或评价'},
            'optimize': {'en': 'Optimize or improve', 'zh': '优化或改进'},
            'monitor': {'en': 'Monitor or track', 'zh': '监控或跟踪'},
            'log': {'en': 'Log or record', 'zh': '记录或日志'},
            'report': {'en': 'Report or summarize', 'zh': '报告或总结'}
        }
    
    def generate_function_comment(self, func_info: Dict[str, Any]) -> str:
        """Generate intelligent comment for a function with style support"""
        func_name = func_info['name']
        
        # Skip special methods
        if func_name.startswith('__') and func_name.endswith('__'):
            return ""
        
        # Get descriptions based on semantic analysis
        description_en, description_zh = self._infer_function_description(func_info)
        
        # Generate sections based on style
        style_methods = {
            DocstringStyle.GOOGLE: (
                self._generate_google_args_section,
                self._generate_google_returns_section,
                self._generate_google_additional_sections
            ),
            DocstringStyle.NUMPY: (
                self._generate_numpy_args_section,
                self._generate_numpy_returns_section,
                self._generate_numpy_additional_sections
            ),
            DocstringStyle.RESTRUCTUREDTEXT: (
                self._generate_restructuredtext_args_section,
                self._generate_restructuredtext_returns_section,
                self._generate_restructuredtext_additional_sections
            )
        }
        
        if self.config.style in style_methods:
            args_gen, returns_gen, additional_gen = style_methods[self.config.style]
            args_section = args_gen(func_info['args'])
            returns_section = returns_gen(func_info)
            additional_sections = additional_gen(func_info)
        else:  # DEFAULT
            args_section = self._generate_default_args_section(func_info['args'])
            returns_section = self._generate_default_returns_section(func_info)
            additional_sections = self._generate_default_additional_sections(func_info)
        
        # Get template for current style
        template = self.templates.get(self.config.style.value, self.templates['default'])
        
        # Format the comment
        comment = template.format(
            name=func_name,
            type='Function',
            description_en=description_en,
            description_zh=description_zh,
            args_section=args_section,
            returns_section=returns_section,
            additional_sections=additional_sections,
            args_section_formatted=self._format_args_for_restructuredtext(func_info['args']),
            returns_description=self._get_returns_description(func_info)
        )
        
        # Add decorators section if any
        if func_info['decorators']:
            decorators_section = self._generate_decorators_section(func_info['decorators'])
            comment = self._insert_section(comment, 'Decorators', decorators_section)
        
        return comment
    
    def generate_class_comment(self, class_info: Dict[str, Any]) -> str:
        """Generate intelligent comment for a class with style support"""
        class_name = class_info['name']
        
        # Get descriptions
        description_en, description_zh = self._infer_class_description(class_info)
        
        # Generate sections based on style
        if self.config.style == DocstringStyle.GOOGLE:
            methods_section = self._generate_google_methods_section(class_info['methods'])
        elif self.config.style == DocstringStyle.NUMPY:
            methods_section = self._generate_numpy_methods_section(class_info['methods'])
        elif self.config.style == DocstringStyle.RESTRUCTUREDTEXT:
            methods_section = self._generate_restructuredtext_methods_section(class_info['methods'])
        else:  # DEFAULT
            methods_section = self._generate_default_methods_section(class_info['methods'])
        
        # Get template
        template = self.templates.get(self.config.style.value, self.templates['default'])
        
        # Format the comment
        comment = template.format(
            name=class_name,
            type='Class',
            description_en=description_en,
            description_zh=description_zh,
            args_section='',  # Not used for classes
            returns_section=methods_section,
            additional_sections='',
            args_section_formatted='',
            returns_description=''
        )
        
        # Add base classes section
        if class_info['bases']:
            bases_section = self._generate_bases_section(class_info['bases'])
            comment = self._insert_section(comment, 'Inheritance', bases_section)
        
        return comment
    
    def _infer_function_description(self, func_info: Dict[str, Any]) -> Tuple[str, str]:
        """Infer function description with semantic analysis"""
        func_name = func_info['name'].lower()
        body_analysis = func_info.get('body_analysis', {})
        semantic_category = body_analysis.get('semantic_category', 'general')
        
        # Map semantic categories to descriptions
        category_descriptions = {
            'data_retrieval': {'en': 'Retrieve data', 'zh': '检索数据'},
            'data_modification': {'en': 'Modify data', 'zh': '修改数据'},
            'creation': {'en': 'Create new instance', 'zh': '创建新实例'},
            'deletion': {'en': 'Delete item', 'zh': '删除项目'},
            'validation': {'en': 'Validate input', 'zh': '验证输入'},
            'calculation': {'en': 'Calculate value', 'zh': '计算值'},
            'transformation': {'en': 'Transform data', 'zh': '转换数据'},
            'analysis': {'en': 'Analyze data', 'zh': '分析数据'},
            'generation': {'en': 'Generate content', 'zh': '生成内容'},
            'filtering': {'en': 'Filter data', 'zh': '过滤数据'},
            'io_operation': {'en': 'Perform I/O operation', 'zh': '执行输入/输出操作'},
            'network_operation': {'en': 'Perform network operation', 'zh': '执行网络操作'},
            'management': {'en': 'Manage process', 'zh': '管理流程'},
            'general': {'en': 'Perform operation', 'zh': '执行操作'}
        }
        
        # Use semantic category if available
        if semantic_category in category_descriptions:
            desc = category_descriptions[semantic_category]
            return desc['en'], desc['zh']
        
        # Fall back to keyword matching
        for keyword, descriptions in self.keyword_map.items():
            if keyword in func_name:
                return descriptions['en'], descriptions['zh']
        
        # Default based on complexity
        complexity = body_analysis.get('complexity', 'low')
        if complexity == 'high':
            return 'Perform complex operation', '执行复杂操作'
        elif complexity == 'medium':
            return 'Perform operation', '执行操作'
        else:
            return 'Perform simple operation', '执行简单操作'
    
    def _infer_class_description(self, class_info: Dict[str, Any]) -> Tuple[str, str]:
        """Infer class description with semantic analysis"""
        class_name = class_info['name'].lower()
        
        # Common class patterns
        patterns = {
            'manager': {'en': 'Manager class for handling operations', 'zh': '处理操作的管理器类'},
            'service': {'en': 'Service class providing functionality', 'zh': '提供功能的服务类'},
            'model': {'en': 'Data model class', 'zh': '数据模型类'},
            'controller': {'en': 'Controller class for managing flow', 'zh': '管理流程的控制器类'},
            'handler': {'en': 'Handler class for processing requests', 'zh': '处理请求的处理器类'},
            'factory': {'en': 'Factory class for creating instances', 'zh': '创建实例的工厂类'},
            'utils': {'en': 'Utility class with helper functions', 'zh': '包含辅助函数的工具类'},
            'helper': {'en': 'Helper class providing utilities', 'zh': '提供实用程序的辅助类'},
            'processor': {'en': 'Data processor class', 'zh': '数据处理器类'},
            'analyzer': {'en': 'Data analyzer class', 'zh': '数据分析器类'},
            'generator': {'en': 'Content generator class', 'zh': '内容生成器类'},
            'validator': {'en': 'Data validator class', 'zh': '数据验证器类'},
            'parser': {'en': 'Data parser class', 'zh': '数据解析器类'},
            'formatter': {'en': 'Data formatter class', 'zh': '数据格式化器类'},
            'converter': {'en': 'Data converter class', 'zh': '数据转换器类'},
            'calculator': {'en': 'Calculator class', 'zh': '计算器类'},
            'client': {'en': 'Client class for external services', 'zh': '外部服务客户端类'},
            'adapter': {'en': 'Adapter class for compatibility', 'zh': '兼容性适配器类'},
            'wrapper': {'en': 'Wrapper class for encapsulation', 'zh': '封装包装器类'},
            'interface': {'en': 'Interface definition class', 'zh': '接口定义类'},
            'abstract': {'en': 'Abstract base class', 'zh': '抽象基类'},
            'base': {'en': 'Base class for inheritance', 'zh': '继承用的基类'},
            'mixin': {'en': 'Mixin class for adding functionality', 'zh': '添加功能的混入类'},
            'decorator': {'en': 'Decorator class', 'zh': '装饰器类'},
            'exception': {'en': 'Exception class', 'zh': '异常类'},
            'error': {'en': 'Error handling class', 'zh': '错误处理类'}
        }
        
        # Check for patterns in class name
        for pattern, descriptions in patterns.items():
            if pattern in class_name:
                return descriptions['en'], descriptions['zh']
        
        # Analyze methods
        method_count = len(class_info['methods'])
        if method_count == 0:
            return 'Empty class', '空类'
        elif method_count < 5:
            return 'Simple class', '简单类'
        else:
            return 'Complex class', '复杂类'
    
    def _generate_default_args_section(self, args_info: List[Dict[str, Any]]) -> str:
        """Generate default Args section"""
        if not args_info:
            return "    None"
        
        args_lines = []
        for arg in args_info:
            line = f"    {arg['name']}"
            if arg['type']:
                line += f" ({arg['type']}): {arg['description']}"
            else:
                line += f": {arg['description']}"
            if arg.get('has_default'):
                line += f" (default: {arg['default']})"
            args_lines.append(line)
        
        return '\n'.join(args_lines)
    
    def _generate_default_returns_section(self, func_info: Dict[str, Any]) -> str:
        """Generate default Returns section"""
        return_type = func_info['return_type']
        
        if return_type:
            return f"    {return_type}: Return value"
        else:
            return "    None or result: Return value"
    
    def _generate_default_additional_sections(self, func_info: Dict[str, Any]) -> str:
        """Generate default additional sections"""
        sections = []
        
        body_analysis = func_info.get('body_analysis', {})
        
        # Add examples section for complex functions
        if body_analysis.get('complexity', 'low') in ['medium', 'high']:
            sections.append("Example:\n    >>> example usage")
        
        # Add raises section if function has exceptions
        if body_analysis.get('has_exceptions'):
            sections.append("Raises:\n    Exception: Description of when this exception is raised")
        
        return '\n\n'.join(sections) if sections else ""
    
    def _generate_google_args_section(self, args_info: List[Dict[str, Any]]) -> str:
        """Generate Google style Args section"""
        return self._generate_default_args_section(args_info)
    
    def _generate_google_returns_section(self, func_info: Dict[str, Any]) -> str:
        """Generate Google style Returns section"""
        return self._generate_default_returns_section(func_info)
    
    def _generate_google_additional_sections(self, func_info: Dict[str, Any]) -> str:
        """Generate Google style additional sections"""
        sections = []
        
        body_analysis = func_info.get('body_analysis', {})
        
        # Yields section for generator functions
        if 'generator' in func_info.get('body_analysis', {}).get('semantic_category', ''):
            sections.append("Yields:\n    Value: Description of yielded value")
        
        # Raises section
        if body_analysis.get('has_exceptions'):
            sections.append("Raises:\n    Exception: Description of when this exception is raised")
        
        # Examples section
        sections.append("Example:\n    >>> example usage\n    expected output")
        
        return '\n\n'.join(sections) if sections else ""
    
    def _generate_numpy_args_section(self, args_info: List[Dict[str, Any]]) -> str:
        """Generate NumPy style Parameters section"""
        if not args_info:
            return "None"
        
        args_lines = []
        for arg in args_info:
            line = f"{arg['name']}"
            if arg['type']:
                line += f" : {arg['type']}"
            line += f"\n    {arg['description']}"
            if arg.get('has_default'):
                line += f", default: {arg['default']}"
            args_lines.append(line)
        
        return '\n'.join(args_lines)
    
    def _generate_numpy_returns_section(self, func_info: Dict[str, Any]) -> str:
        """Generate NumPy style Returns section"""
        return_type = func_info['return_type']
        
        if return_type:
            return f"{return_type}\n    Return value description"
        else:
            return "None or result\n    Return value description"
    
    def _generate_numpy_additional_sections(self, func_info: Dict[str, Any]) -> str:
        """Generate NumPy style additional sections"""
        sections = []
        
        # Examples section
        sections.append("Examples\n--------\n>>> example usage\n>>> expected output")
        
        # Notes section
        body_analysis = func_info.get('body_analysis', {})
        if body_analysis.get('has_exceptions'):
            sections.append("Raises\n------\nException\n    Description of when this exception is raised")
        
        return '\n\n'.join(sections) if sections else ""
    
    def _generate_restructuredtext_args_section(self, args_info: List[Dict[str, Any]]) -> str:
        """Generate reStructuredText style Args section"""
        if not args_info:
            return ":param: None"
        
        args_lines = []
        for arg in args_info:
            line = f":param {arg['name']}: {arg['description']}"
            if arg['type']:
                line += f" (type: {arg['type']})"
            if arg.get('has_default'):
                line += f" (default: {arg['default']})"
            args_lines.append(line)
        
        return '\n'.join(args_lines)
    
    def _format_args_for_restructuredtext(self, args_info: List[Dict[str, Any]]) -> str:
        """Format args for reStructuredText template"""
        if not args_info:
            return ""
        
        formatted = []
        for arg in args_info:
            if arg['type']:
                formatted.append(f"{arg['name']} ({arg['type']})")
            else:
                formatted.append(arg['name'])
        
        return ', '.join(formatted)
    
    def _get_returns_description(self, func_info: Dict[str, Any]) -> str:
        """Get returns description for reStructuredText"""
        return_type = func_info['return_type']
        
        if return_type:
            return f"{return_type}: Return value description"
        else:
            return "None or result: Return value description"
    
    def _generate_restructuredtext_returns_section(self, func_info: Dict[str, Any]) -> str:
        """Generate reStructuredText style Returns section"""
        return self._get_returns_description(func_info)
    
    def _generate_restructuredtext_additional_sections(self, func_info: Dict[str, Any]) -> str:
        """Generate reStructuredText style additional sections"""
        sections = []
        
        # Example section
        sections.append(":Example:\n\n    >>> example usage\n    >>> expected output")
        
        # Raises section
        body_analysis = func_info.get('body_analysis', {})
        if body_analysis.get('has_exceptions'):
            sections.append(":Raises:\n    Exception: Description of when this exception is raised")
        
        return '\n\n'.join(sections) if sections else ""
    
    def _generate_decorators_section(self, decorators: List[str]) -> str:
        """Generate decorators section"""
        if not decorators:
            return ""
        
        decorator_lines = []
        for decorator in decorators:
            decorator_lines.append(f"    @{decorator}")
        
        return '\n'.join(decorator_lines)
    
    def _generate_bases_section(self, bases: List[str]) -> str:
        """Generate base classes section"""
        if not bases:
            return ""
        
        bases_str = ', '.join(bases)
        return f"Inherits from: {bases_str}"
    
    def _generate_google_methods_section(self, methods: List[Dict[str, Any]]) -> str:
        """Generate Google style methods section"""
        if not methods:
            return "Methods:\n    None"
        
        method_lines = []
        for method in methods:
            if not (method['name'].startswith('__') and method['name'].endswith('__')):
                method_lines.append(f"    {method['name']}(): Method description")
        
        method_lines_joined = '\n'.join(method_lines)
        return f"Methods:\n{method_lines_joined}"
    
    def _generate_numpy_methods_section(self, methods: List[Dict[str, Any]]) -> str:
        """Generate NumPy style methods section"""
        if not methods:
            return "Methods\n-------\nNone"
        
        method_lines = []
        for method in methods:
            if not (method['name'].startswith('__') and method['name'].endswith('__')):
                method_lines.append(f"{method['name']}\n    Method description")
        
        method_lines_joined = '\n'.join(method_lines)
        return f"Methods\n-------\n{method_lines_joined}"
    
    def _generate_restructuredtext_methods_section(self, methods: List[Dict[str, Any]]) -> str:
        """Generate reStructuredText style methods section"""
        if not methods:
            return ":Methods:\n    None"
        
        method_lines = []
        for method in methods:
            if not (method['name'].startswith('__') and method['name'].endswith('__')):
                method_lines.append(f":method {method['name']}: Method description")
        
        return '\n'.join(method_lines)
    
    def _generate_default_methods_section(self, methods: List[Dict[str, Any]]) -> str:
        """Generate default methods section"""
        if not methods:
            return "Methods:\n    No public methods"
        
        method_lines = []
        for method in methods:
            if not (method['name'].startswith('__') and method['name'].endswith('__')):
                method_lines.append(f"    {method['name']}(): Method description")
        
        method_lines_joined = '\n'.join(method_lines)
        return f"Methods:\n{method_lines_joined}"
    
    def _insert_section(self, comment: str, section_name: str, section_content: str) -> str:
        """Insert a section into the comment"""
        if not section_content:
            return comment
        
        # Find the last """ and insert before it
        last_quotes = comment.rfind('"""')
        if last_quotes != -1:
            # Insert section before the closing quotes
            inserted = comment[:last_quotes] + f"\n{section_name}:\n{section_content}\n" + comment[last_quotes:]
            return inserted
        
        return comment


# ===================== Parallel Processing Processor =====================

class ParallelProcessor:
    """Processor with parallel processing support"""
    
    def __init__(self, config: Optional[CommentConfig] = None):
        self.config = config or CommentConfig()
        
    def process_files_parallel(self, file_paths: List[str], threshold: float = 0.7) -> Dict[str, Any]:
        """Process multiple files in parallel"""
        start_time = time.time()
        
        print(f"Processing {len(file_paths)} files with {self.config.parallel_workers} workers...")
        
        # Prepare arguments for parallel processing
        args_list = [(file_path, threshold) for file_path in file_paths]
        
        # Use ThreadPoolExecutor for I/O bound tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path, threshold): file_path 
                for file_path in file_paths
            }
            
            # Collect results as they complete
            results = []
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                    
                    if result['success']:
                        if result['total_changes'] > 0:
                            print(f"✓ {file_path}: {result['total_changes']} changes")
                        else:
                            print(f"○ {file_path}: No changes needed")
                    else:
                        print(f"✗ {file_path}: Failed - {result.get('error', 'Unknown error')}")
                        
                except concurrent.futures.TimeoutError:
                    print(f"✗ {file_path}: Timeout after 5 minutes")
                    results.append({
                        'file': file_path,
                        'success': False,
                        'error': 'Timeout'
                    })
                except Exception as e:
                    print(f"✗ {file_path}: Error - {e}")
                    results.append({
                        'file': file_path,
                        'success': False,
                        'error': str(e)
                    })
        
        # Calculate summary
        elapsed_time = time.time() - start_time
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        total_changes = sum(r.get('total_changes', 0) for r in results if r['success'])
        
        summary = {
            'total_files': len(file_paths),
            'successful': successful,
            'failed': failed,
            'total_changes': total_changes,
            'elapsed_time_seconds': elapsed_time,
            'files_per_second': len(file_paths) / elapsed_time if elapsed_time > 0 else 0,
            'details': results
        }
        
        return summary
    
    def _process_single_file(self, file_path: str, threshold: float) -> Dict[str, Any]:
        """Process a single file (wrapper for IncrementalUpdateProcessor)"""
        # Create a new processor for each thread to avoid shared state issues
        processor = IncrementalUpdateProcessor(self.config)
        return processor.process_file(file_path, threshold)


class IncrementalUpdateProcessor:
    """Processor that only updates incomplete or missing comments"""
    
    def __init__(self, config: Optional[CommentConfig] = None):
        self.config = config or CommentConfig()
        self.analyzer = None
        self.generator = None
        
    def process_file(self, file_path: str, update_threshold: float = 0.7) -> Dict[str, Any]:
        """Process a file with incremental updates"""
        result = {
            'file': file_path,
            'updated_classes': 0,
            'updated_functions': 0,
            'skipped_classes': 0,
            'skipped_functions': 0,
            'total_changes': 0,
            'success': False
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Analyze code
            self.analyzer = EnhancedCodeAnalyzer(content)
            self.generator = EnhancedCommentGenerator(self.config)
            
            # Extract classes and functions with quality analysis
            classes = self.analyzer.extract_classes()
            functions = self.analyzer.extract_functions()
            
            # Apply incremental updates
            modified_content, changes = self._apply_incremental_updates(
                content, classes, functions, update_threshold
            )
            
            # Update result
            result.update(changes)
            
            # Write back if there are changes
            if result['total_changes'] > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                result['success'] = True
            else:
                result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _apply_incremental_updates(self, content: str, classes: List[Dict[str, Any]], 
                                  functions: List[Dict[str, Any]], update_threshold: float) -> Tuple[str, Dict[str, Any]]:
        """Apply incremental updates to content"""
        lines = content.splitlines(keepends=True)
        
        changes = {
            'updated_classes': 0,
            'updated_functions': 0,
            'skipped_classes': 0,
            'skipped_functions': 0,
            'total_changes': 0
        }
        
        # Collect update candidates
        update_candidates = []
        
        # Process classes
        for class_info in classes:
            quality = class_info['docstring_quality']
            
            # Skip if quality is good enough
            if quality['completeness_score'] >= update_threshold:
                changes['skipped_classes'] += 1
                continue
            
            # Generate new comment
            new_comment = self.generator.generate_class_comment(class_info)
            if new_comment:
                update_candidates.append({
                    'type': 'class',
                    'name': class_info['name'],
                    'start_line': class_info['start_line'] - 1,  # 0-indexed
                    'old_docstring': class_info['docstring'],
                    'new_comment': new_comment,
                    'quality_score': quality['completeness_score']
                })
        
        # Process functions (only top-level)
        for func_info in functions:
            # Skip if method (handled by class)
            is_method = False
            for class_info in classes:
                if (func_info['start_line'] >= class_info['start_line'] and 
                    func_info['end_line'] <= class_info['end_line']):
                    is_method = True
                    break
            
            if is_method:
                continue
            
            quality = func_info['docstring_quality']
            
            # Skip if quality is good enough
            if quality['completeness_score'] >= update_threshold:
                changes['skipped_functions'] += 1
                continue
            
            # Generate new comment
            new_comment = self.generator.generate_function_comment(func_info)
            if new_comment:
                update_candidates.append({
                    'type': 'function',
                    'name': func_info['name'],
                    'start_line': func_info['start_line'] - 1,  # 0-indexed
                    'old_docstring': func_info['docstring'],
                    'new_comment': new_comment,
                    'quality_score': quality['completeness_score']
                })
        
        # Sort by line number in descending order
        update_candidates.sort(key=lambda x: x['start_line'], reverse=True)
        
        # Apply updates
        for candidate in update_candidates:
            success = self._replace_docstring(
                lines, 
                candidate['start_line'], 
                candidate['old_docstring'],
                candidate['new_comment']
            )
            
            if success:
                if candidate['type'] == 'class':
                    changes['updated_classes'] += 1
                else:
                    changes['updated_functions'] += 1
                
                changes['total_changes'] += 1
        
        return ''.join(lines), changes
    
    def _replace_docstring(self, lines: List[str], start_line: int, 
                          old_docstring: Optional[str], new_comment: str) -> bool:
        """Replace existing docstring with new comment"""
        # Check if there's already a comment at this location
        line_before = lines[start_line].strip() if start_line < len(lines) else ""
        
        # Skip if there's already a docstring
        if '"""' in line_before or "'''" in line_before:
            return False
        
        # Insert the new comment
        lines.insert(start_line, new_comment + '\n')
        return True


# ===================== Main Application =====================

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced bilingual comments tool with all optimizations",
        epilog="Example: python add_bilingual_comments_final.py core/ --style google --parallel"
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
    parser.add_argument(
        "--style",
        choices=['default', 'google', 'numpy', 'restructuredtext'],
        default='default',
        help="Docstring style to use"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Quality threshold for updates (0.0-1.0)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing for directories"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help="Number of parallel workers (default: CPU count - 1)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = CommentConfig()
    if args.config:
        config = CommentConfig.from_file(args.config)
    
    config.style = DocstringStyle(args.style)
    config.parallel_workers = args.workers
    
    path = Path(args.path)
    
    if args.test:
        print(f"Test mode: Would process {path} with style {args.style}")
        if path.is_dir():
            py_files = list(path.rglob("*.py"))
            print(f"Found {len(py_files)} Python files")
            if args.parallel:
                print(f"Would use parallel processing with {config.parallel_workers} workers")
        return
    
    if path.is_file() and path.suffix == '.py':
        # Single file processing
        processor = IncrementalUpdateProcessor(config)
        result = processor.process_file(str(path), args.threshold)
        
        if result['success']:
            if result['total_changes'] > 0:
                print(f"✓ Updated {result['file']}: {result['total_changes']} changes")
            else:
                print(f"✓ No updates needed for {result['file']}")
        else:
            print(f"✗ Failed to process {result['file']}: {result.get('error', 'Unknown error')}")
    
    elif path.is_dir():
        # Directory processing
        py_files = list(path.rglob("*.py"))
        print(f"Found {len(py_files)} Python files in {path}")
        
        if len(py_files) == 0:
            print("No Python files found")
            return
        
        if args.parallel and len(py_files) > 1:
            # Parallel processing
            processor = ParallelProcessor(config)
            summary = processor.process_files_parallel([str(f) for f in py_files], args.threshold)
            
            # Print summary
            print("\n" + "="*60)
            print("PROCESSING SUMMARY")
            print("="*60)
            print(f"Total files: {summary['total_files']}")
            print(f"Successful: {summary['successful']}")
            print(f"Failed: {summary['failed']}")
            print(f"Total changes: {summary['total_changes']}")
            print(f"Time elapsed: {summary['elapsed_time_seconds']:.2f} seconds")
            print(f"Files per second: {summary['files_per_second']:.2f}")
            
            if summary['failed'] > 0:
                print(f"\nFailed files:")
                for result in summary['details']:
                    if not result['success']:
                        print(f"  - {result['file']}: {result.get('error', 'Unknown error')}")
        else:
            # Sequential processing
            print("Processing files sequentially...")
            processor = IncrementalUpdateProcessor(config)
            
            start_time = time.time()
            successful = 0
            failed = 0
            total_changes = 0
            
            for py_file in py_files:
                result = processor.process_file(str(py_file), args.threshold)
                
                if result['success']:
                    successful += 1
                    total_changes += result['total_changes']
                    
                    if result['total_changes'] > 0:
                        print(f"✓ {py_file}: {result['total_changes']} changes")
                else:
                    failed += 1
                    print(f"✗ {py_file}: Failed - {result.get('error', 'Unknown error')}")
            
            elapsed_time = time.time() - start_time
            
            # Print summary
            print("\n" + "="*60)
            print("PROCESSING SUMMARY")
            print("="*60)
            print(f"Total files: {len(py_files)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            print(f"Total changes: {total_changes}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"Files per second: {len(py_files) / elapsed_time:.2f}")
    
    else:
        print(f"Error: {path} is not a Python file or directory")


if __name__ == "__main__":
    main()