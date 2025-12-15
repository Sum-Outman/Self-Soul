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

# Models package initialization file
# Import all core models (延迟导入以避免循环依赖)
from .manager import ManagerModel
from .language import LanguageModel
from .audio import AudioProcessingModel
from .vision import VisionModel
from .video import VideoVisionModel
from .spatial import SpatialPerceptionModel
from .sensor import SensorPerceptionModel
from .computer import ComputerModel
from .motion import MotionModel
from .knowledge import KnowledgeModel
from .programming import ProgrammingModel
from .planning import PlanningModel

# 延迟模型注册，避免循环依赖
# 模型将在首次使用时通过model_registry动态注册
def register_all_models():
    """延迟注册所有模型到注册表"""
    try:
        from core.model_registry import get_model_registry
        model_registry = get_model_registry()
        
        # 注册所有模型到注册表
        model_registry.register_model("manager", ManagerModel)
        model_registry.register_model("language", LanguageModel)
        model_registry.register_model("audio", AudioProcessingModel)
        model_registry.register_model("vision_image", VisionModel)
        model_registry.register_model("vision_video", VideoVisionModel)
        model_registry.register_model("spatial", SpatialPerceptionModel)
        model_registry.register_model("sensor", SensorPerceptionModel)
        model_registry.register_model("computer", ComputerModel)
        model_registry.register_model("motion", MotionModel)
        model_registry.register_model("knowledge", KnowledgeModel)
        model_registry.register_model("programming", ProgrammingModel)
        model_registry.register_model("planning", PlanningModel)
        
        return True
    except Exception as e:
        print(f"模型注册失败: {e}")
        return False

# 导出所有模型类
__all__ = [
    'ManagerModel',
    'LanguageModel', 
    'AudioProcessingModel',
    'VisionModel',
    'VideoVisionModel',
    'SpatialPerceptionModel',
    'SensorPerceptionModel',
    'ComputerModel',
    'MotionModel',
    'KnowledgeModel',
    'ProgrammingModel',
    'PlanningModel'
]
