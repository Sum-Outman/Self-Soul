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

#!/usr/bin/env python
# 启动Self Soul 的入口脚本

import os
import sys

# 添加项目根目录到Python路径（插入到最前面以确保优先使用我们的six.py）
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 强制导入我们的six.py并替换sys.modules中的six模块
import six as original_six
# 删除现有的six模块引用
if 'six' in sys.modules:
    del sys.modules['six']
# 导入我们的six.py
import six
# 确保我们的six模块被正确设置
sys.modules['six'] = six

# 添加core目录到Python路径
core_path = os.path.join(project_root, 'core')
sys.path.append(core_path)

# 添加models目录到Python路径
models_path = os.path.join(core_path, 'models')
sys.path.append(models_path)

# 导入并运行协调器
from core.agi_coordinator import AGICoordinator

# 创建Self Soul 实例
agi_system = AGICoordinator()

if __name__ == "__main__":
    # 示例使用
    # 测试中文文本输入
    print(agi_system.process_user_input('你好，世界', 'text', 'zh'))
    
    # 测试英文文本输入
    print(agi_system.process_user_input('Hello world', 'text', 'en'))
    
    # 测试任务协调
    print(agi_system.coordinate_task('请分析这张图片内容'))
    
    # 测试系统状态
    print(agi_system.get_system_status())
