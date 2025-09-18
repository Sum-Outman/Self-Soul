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

import os
import shutil

# 备份核心目录
shutil.move('core', 'core_backup')

# 创建新的核心目录结构
os.makedirs('core/models', exist_ok=True)

# 移动模型文件
shutil.move('core_backup/models', 'core/models')

# 移动协调器文件
shutil.move('core_backup/coordinator.py', 'core/coordinator.py')

# 清理备份目录
shutil.rmtree('core_backup')

print("项目结构重构完成")