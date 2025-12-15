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
# Self Brain 统一错误处理和日志系统
AGI System Unified Error Handling and Logging System
"""
import logging
import traceback
import sys
from datetime import datetime
import os

# 确保日志目录存在
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'agi_system_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)


"""
ErrorHandler类 - 中文类描述
ErrorHandler Class - English class description
"""
class ErrorHandler:
    """Self Brain 统一错误处理类"""
    
    def __init__(self):
        self.logger = logging.getLogger('AGI_System')
        
    def initialize(self):
        """初始化错误处理器
        Initialize error handler
        
        Returns:
            dict: 初始化结果
        """
        self.logger.info("AGI错误处理器初始化成功")
        return {"success": True, "message": "AGI错误处理器初始化成功"}
        
    def handle_error(self, error, component_name, details=None):
        """处理系统错误"""
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # 记录错误日志
        self.logger.error(f"[{component_name}] {error_type}: {error_message}")
        self.logger.debug(f"Stack trace: {stack_trace}")
        if details:
            self.logger.debug(f"Additional details: {details}")
        
        # 根据错误类型返回不同级别的响应
        if isinstance(error, ValueError):
            return {
                'success': False,
                'error_type': 'validation',
                'message': error_message
            }
        elif isinstance(error, ConnectionError):
            return {
                'success': False,
                'error_type': 'connection',
                'message': '无法连接到外部服务，请检查网络连接'
            }
        elif isinstance(error, FileNotFoundError):
            return {
                'success': False,
                'error_type': 'file_not_found',
                'message': f'文件未找到: {error_message}'
            }
        else:
            return {
                'success': False,
                'error_type': 'internal',
                'message': '系统内部错误，请联系管理员silencecrowtom@qq.com'
            }
        
    def log_info(self, message, component_name, details=None):
        """记录信息日志"""
        self.logger.info(f"[{component_name}] {message}")
        if details:
            self.logger.debug(f"Details: {details}")
        
    def log_warning(self, message, component_name, details=None):
        """记录警告日志"""
        self.logger.warning(f"[{component_name}] {message}")
        if details:
            self.logger.debug(f"Details: {details}")
            
    def log_error(self, message, component_name, details=None):
        """记录错误日志"""
        self.logger.error(f"[{component_name}] {message}")
        if details:
            self.logger.debug(f"Details: {details}")

# 创建全局错误处理器实例
error_handler = ErrorHandler()
