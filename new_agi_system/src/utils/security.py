"""
安全系统

提供输入验证、安全检查和威胁检测功能。
"""

import re
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import base64

logger = logging.getLogger(__name__)


class SecurityValidator:
    """安全验证器"""
    
    def __init__(self):
        # 安全配置
        self.config = {
            'max_input_length': 10000,
            'max_json_depth': 10,
            'allowed_special_chars': r'[!@#$%^&*()_+\-=\[\]{};:"\\|,.<>/?]',
            'blocked_patterns': [
                r'<script[^>]*>.*?</script>',  # XSS脚本
                r'on\w+\s*=',  # 事件处理器
                r'javascript:',  # JavaScript协议
                r'vbscript:',  # VBScript协议
                r'expression\s*\(',  # CSS表达式
                r'--\s',  # SQL注释
                r'union\s+select',  # SQL注入
                r'exec\s*\(',  # 命令执行
                r'fromcharcode\s*\(',  # 字符编码绕过
                r'eval\s*\(',  # JavaScript eval
                r'import\s+',  # Python导入
                r'__import__\s*\(',  # Python动态导入
                r'open\s*\(',  # 文件操作
                r'read\s*\(',  # 文件读取
                r'write\s*\(',  # 文件写入
                r'subprocess\s*',  # 子进程
                r'os\s*\.',  # 操作系统调用
                r'sys\s*\.',  # 系统调用
            ],
            'rate_limit_requests': 100,  # 每分钟请求数限制
            'rate_limit_window': 60,  # 时间窗口（秒）
        }
        
        # 编译阻塞模式
        self.blocked_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                for pattern in self.config['blocked_patterns']]
        
        # 速率限制跟踪
        self.request_counts = {}
        
        logger.info("安全验证器已初始化")
    
    def validate_input(self, input_data: Any, input_type: str = 'general') -> Tuple[bool, str]:
        """
        验证输入数据
        
        参数:
            input_data: 输入数据
            input_type: 输入类型 ('text', 'json', 'command', 'general')
        
        返回:
            (是否有效, 错误消息)
        """
        try:
            if input_data is None:
                return True, "输入为空"
            
            # 基于类型进行验证
            if input_type == 'text':
                return self._validate_text(input_data)
            elif input_type == 'json':
                return self._validate_json(input_data)
            elif input_type == 'command':
                return self._validate_command(input_data)
            else:
                return self._validate_general(input_data)
                
        except Exception as e:
            logger.error(f"输入验证错误: {e}")
            return False, f"验证错误: {str(e)}"
    
    def _validate_text(self, text: str) -> Tuple[bool, str]:
        """验证文本输入"""
        if not isinstance(text, str):
            return False, "输入不是字符串"
        
        # 检查长度
        if len(text) > self.config['max_input_length']:
            return False, f"文本过长 (最大 {self.config['max_input_length']} 字符)"
        
        # 检查阻塞模式
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                return False, f"检测到不安全内容: {pattern.pattern}"
        
        # 检查特殊字符比例（防止编码攻击）
        special_chars = re.findall(self.config['allowed_special_chars'], text)
        if len(special_chars) > len(text) * 0.3:  # 特殊字符超过30%
            return False, "特殊字符比例过高"
        
        return True, "文本验证通过"
    
    def _validate_json(self, json_data: Any) -> Tuple[bool, str]:
        """验证JSON输入"""
        try:
            # 如果是字符串，先解析
            if isinstance(json_data, str):
                parsed = json.loads(json_data)
                # 检查递归深度
                depth = self._calculate_json_depth(parsed)
                if depth > self.config['max_json_depth']:
                    return False, f"JSON深度过大 (最大 {self.config['max_json_depth']} 层)"
            else:
                parsed = json_data
            
            # 转换为字符串进行检查
            json_str = json.dumps(parsed)
            return self._validate_text(json_str)
            
        except json.JSONDecodeError as e:
            return False, f"JSON解析错误: {str(e)}"
        except Exception as e:
            return False, f"JSON验证错误: {str(e)}"
    
    def _validate_command(self, command: str) -> Tuple[bool, str]:
        """验证命令输入"""
        if not isinstance(command, str):
            return False, "命令不是字符串"
        
        # 命令白名单模式（简化版）
        safe_patterns = [
            r'^[a-zA-Z0-9_\-\s\.\/]+$',  # 仅允许字母数字、下划线、连字符、点、斜杠
        ]
        
        for pattern in safe_patterns:
            if not re.match(pattern, command):
                return False, f"命令包含不安全字符"
        
        # 检查阻塞模式
        for pattern in self.blocked_patterns:
            if pattern.search(command):
                return False, f"检测到不安全命令: {pattern.pattern}"
        
        return True, "命令验证通过"
    
    def _validate_general(self, data: Any) -> Tuple[bool, str]:
        """通用验证"""
        if isinstance(data, str):
            return self._validate_text(data)
        elif isinstance(data, (dict, list)):
            return self._validate_json(data)
        elif isinstance(data, (int, float, bool)):
            return True, "基本类型验证通过"
        else:
            return False, f"不支持的数据类型: {type(data)}"
    
    def _calculate_json_depth(self, data: Any, current_depth: int = 0) -> int:
        """计算JSON结构深度"""
        if isinstance(data, dict):
            if not data:
                return current_depth + 1
            return max(self._calculate_json_depth(value, current_depth + 1) 
                      for value in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth + 1
            return max(self._calculate_json_depth(item, current_depth + 1) 
                      for item in data)
        else:
            return current_depth
    
    def check_rate_limit(self, client_id: str) -> Tuple[bool, str]:
        """
        检查速率限制
        
        参数:
            client_id: 客户端标识符
        
        返回:
            (是否允许, 错误消息)
        """
        import time
        current_time = time.time()
        
        # 清理旧记录
        window_start = current_time - self.config['rate_limit_window']
        if client_id in self.request_counts:
            self.request_counts[client_id] = [
                t for t in self.request_counts[client_id] if t > window_start
            ]
        else:
            self.request_counts[client_id] = []
        
        # 检查请求次数
        if len(self.request_counts[client_id]) >= self.config['rate_limit_requests']:
            return False, "请求频率过高，请稍后再试"
        
        # 记录当前请求
        self.request_counts[client_id].append(current_time)
        return True, "速率检查通过"


class InputSanitizer:
    """输入清理器"""
    
    def __init__(self):
        self.escape_patterns = {
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;',
            '(': '&#40;',
            ')': '&#41;',
            '[': '&#91;',
            ']': '&#93;',
            '{': '&#123;',
            '}': '&#125;',
        }
        
        logger.info("输入清理器已初始化")
    
    def sanitize_text(self, text: str) -> str:
        """清理文本输入"""
        if not text:
            return ""
        
        # HTML转义
        sanitized = text
        for char, replacement in self.escape_patterns.items():
            sanitized = sanitized.replace(char, replacement)
        
        # 移除控制字符
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
        
        # 标准化换行符
        sanitized = sanitized.replace('\r\n', '\n').replace('\r', '\n')
        
        # 限制连续空格
        sanitized = re.sub(r' {2,}', ' ', sanitized)
        
        # 限制连续换行
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
        
        return sanitized.strip()
    
    def sanitize_json(self, json_data: Any) -> Any:
        """清理JSON数据"""
        if isinstance(json_data, dict):
            return {self.sanitize_text(str(key)): self.sanitize_json(value) 
                   for key, value in json_data.items()}
        elif isinstance(json_data, list):
            return [self.sanitize_json(item) for item in json_data]
        elif isinstance(json_data, str):
            return self.sanitize_text(json_data)
        else:
            return json_data
    
    def sanitize_html(self, html: str) -> str:
        """清理HTML输入（简化版）"""
        # 移除script标签
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.IGNORECASE | re.DOTALL)
        
        # 移除事件处理器
        sanitized = re.sub(r'on\w+\s*=\s*"[^"]*"', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"on\w+\s*=\s*'[^']*'", '', sanitized, flags=re.IGNORECASE)
        
        # 移除JavaScript协议
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        
        # 允许基本HTML标签（白名单）
        allowed_tags = ['p', 'br', 'b', 'i', 'u', 'strong', 'em', 'ul', 'ol', 'li', 
                       'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span', 'a']
        allowed_attrs = ['href', 'title', 'target', 'class', 'id', 'style']
        
        # 在实际实现中，这里会有更复杂的HTML清理逻辑
        # 现在返回基本清理后的HTML
        return sanitized


class ThreatDetector:
    """威胁检测器"""
    
    def __init__(self):
        # 威胁模式
        self.threat_patterns = {
            'sql_injection': [
                r"'\s*or\s*'",
                r"'\s*and\s*'",
                r"union\s+select",
                r"select\s+\*\s+from",
                r"insert\s+into",
                r"delete\s+from",
                r"update\s+\w+\s+set",
                r"drop\s+table",
                r"create\s+table",
                r"alter\s+table",
                r"exec\s*\(",
                r"xp_cmdshell",
            ],
            'xss': [
                r"<script[^>]*>",
                r"javascript:",
                r"on\w+\s*=",
                r"eval\s*\(",
                r"document\.cookie",
                r"alert\s*\(",
                r"prompt\s*\(",
                r"confirm\s*\(",
                r"<iframe[^>]*>",
                r"<img[^>]*src\s*=",
            ],
            'command_injection': [
                r";\s*\w+",
                r"&\s*\w+",
                r"\|\s*\w+",
                r"`\s*\w+",
                r"\$\s*\(",
                r"cat\s+",
                r"ls\s+",
                r"rm\s+",
                r"wget\s+",
                r"curl\s+",
                r"nc\s+",
                r"ssh\s+",
                r"scp\s+",
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"/etc/passwd",
                r"/etc/shadow",
                r"C:\\Windows\\",
                r"/proc/",
                r"/dev/",
            ]
        }
        
        # 编译模式
        self.compiled_patterns = {}
        for threat_type, patterns in self.threat_patterns.items():
            self.compiled_patterns[threat_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        # 威胁分数阈值
        self.thresholds = {
            'low': 1,
            'medium': 3,
            'high': 5
        }
        
        logger.info("威胁检测器已初始化")
    
    def detect_threats(self, input_data: str) -> Dict[str, Any]:
        """
        检测输入中的威胁
        
        参数:
            input_data: 输入数据
        
        返回:
            威胁检测结果
        """
        if not input_data:
            return {'threat_level': 'none', 'threats': []}
        
        threats = []
        threat_score = 0
        
        # 检查每种威胁类型
        for threat_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(input_data)
                if matches:
                    threat_info = {
                        'type': threat_type,
                        'pattern': pattern.pattern,
                        'matches': matches[:5],  # 限制匹配数量
                        'severity': self._get_severity(threat_type)
                    }
                    threats.append(threat_info)
                    threat_score += threat_info['severity']
        
        # 确定威胁等级
        threat_level = 'none'
        if threat_score >= self.thresholds['high']:
            threat_level = 'high'
        elif threat_score >= self.thresholds['medium']:
            threat_level = 'medium'
        elif threat_score >= self.thresholds['low']:
            threat_level = 'low'
        
        return {
            'threat_level': threat_level,
            'threat_score': threat_score,
            'threats': threats,
            'timestamp': time.time()
        }
    
    def _get_severity(self, threat_type: str) -> int:
        """获取威胁严重程度"""
        severity_map = {
            'sql_injection': 3,
            'xss': 2,
            'command_injection': 4,
            'path_traversal': 3
        }
        return severity_map.get(threat_type, 1)
    
    def is_threatening(self, input_data: str, threshold: str = 'medium') -> bool:
        """
        检查输入是否具有威胁性
        
        参数:
            input_data: 输入数据
            threshold: 威胁阈值 ('low', 'medium', 'high')
        
        返回:
            是否具有威胁性
        """
        result = self.detect_threats(input_data)
        return result['threat_level'] >= threshold