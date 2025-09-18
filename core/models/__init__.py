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

# models 包初始化文件 / Models package initialization file
# 导入所有核心模型 / Import all core models
from .manager.model import ManagerModel
from .language.model import LanguageModel
from .audio.model import AudioProcessingModel
from .vision.merged_model import UnifiedVisionModel
from .vision.video_model import VideoVisionModel
from .spatial.merged_model import SpatialPerceptionModel
from .sensor.model import SensorPerceptionModel
from .computer.model import ComputerModel
from .motion.model import MotionModel
from .knowledge.model import KnowledgeModel
from .programming.model import ProgrammingModel
from .planning.model import PlanningModel

# 导入模型注册表 / Import model registry
from core.model_registry import model_registry

# 注册所有模型到注册表 / Register all models to registry
model_registry.register_model("manager", ManagerModel)
model_registry.register_model("language", LanguageModel)
model_registry.register_model("audio", AudioProcessingModel)
model_registry.register_model("image_vision", UnifiedVisionModel)
model_registry.register_model("video_vision", VideoVisionModel)
model_registry.register_model("spatial", SpatialPerceptionModel)
model_registry.register_model("sensor", SensorPerceptionModel)
model_registry.register_model("computer", ComputerModel)
model_registry.register_model("motion", MotionModel)
model_registry.register_model("knowledge", KnowledgeModel)
model_registry.register_model("programming", ProgrammingModel)
model_registry.register_model("planning", PlanningModel)
