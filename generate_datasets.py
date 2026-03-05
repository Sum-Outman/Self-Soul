import os
import json
import random as random_module

class DeterministicRandom:
    def __init__(self, seed=42):
        self.seed = seed
        self.counter = 0
    
    def _get_hash(self, *args):
        self.counter += 1
        return hash(str(args) + str(self.seed) + str(self.counter))
    
    def choice(self, seq):
        if not seq:
            raise ValueError("Cannot choose from empty sequence")
        return seq[self._get_hash('choice', tuple(seq)) % len(seq)]
    
    def randint(self, a, b):
        return a + self._get_hash('randint', a, b) % (b - a + 1)
    
    def uniform(self, a, b):
        return a + (self._get_hash('uniform', a, b) % 10000) / 10000.0 * (b - a)
    
    def sample(self, population, k):
        # 简化实现
        indices = sorted(range(len(population)), key=lambda i: self._get_hash('sample', i))
        return [population[i] for i in indices[:k]]
    
    def random(self):
        return self.uniform(0.0, 1.0)

# 创建确定性随机实例
random = DeterministicRandom(seed=42)
import time
from datetime import datetime

def generate_datasets():
    """为所有模型类型生成训练数据集"""
    # 定义所有支持的模型类型
    model_types = [
        "manager", "language", "audio", "vision_image", "vision_video",
        "spatial", "sensor", "computer", "motion", "knowledge",
        "programming", "planning", "autonomous", "emotion", "prediction",
        "collaboration", "optimization", "finance", "medical", "value_alignment",
        "stereo_vision"
    ]
    
    # 创建数据集目录
    datasets_dir = "data/datasets"
    os.makedirs(datasets_dir, exist_ok=True)
    
    print("开始为所有模型类型生成训练数据集...")
    
    # 为每种模型类型生成数据集
    for model_type in model_types:
        print(f"\n生成 {model_type} 模型的数据集...")
        
        # 根据模型类型生成数据
        if model_type == "manager":
            data = generate_manager_dataset()
        elif model_type == "language":
            data = generate_language_dataset()
        elif model_type == "audio":
            data = generate_audio_dataset()
        elif model_type == "vision_image":
            data = generate_vision_image_dataset()
        elif model_type == "vision_video":
            data = generate_vision_video_dataset()
        elif model_type == "spatial":
            data = generate_spatial_dataset()
        elif model_type == "sensor":
            data = generate_sensor_dataset()
        elif model_type == "computer":
            data = generate_computer_dataset()
        elif model_type == "motion":
            data = generate_motion_dataset()
        elif model_type == "knowledge":
            data = generate_knowledge_dataset()
        elif model_type == "programming":
            data = generate_programming_dataset()
        elif model_type == "planning":
            data = generate_planning_dataset()
        elif model_type == "autonomous":
            data = generate_autonomous_dataset()
        elif model_type == "emotion":
            data = generate_emotion_dataset()
        elif model_type == "prediction":
            data = generate_prediction_dataset()
        elif model_type == "collaboration":
            data = generate_collaboration_dataset()
        elif model_type == "optimization":
            data = generate_optimization_dataset()
        elif model_type == "finance":
            data = generate_finance_dataset()
        elif model_type == "medical":
            data = generate_medical_dataset()
        elif model_type == "value_alignment":
            data = generate_value_alignment_dataset()
        elif model_type == "stereo_vision":
            data = generate_stereo_vision_dataset()
        else:
            data = generate_default_dataset()
        
        # 保存数据集
        dataset_path = os.path.join(datasets_dir, f"{model_type}_dataset.json")
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"{model_type} 模型数据集已保存到 {dataset_path}")
    
    print("\n所有模型类型的数据集生成完成！")

# 各模型类型的数据集生成函数
def generate_manager_dataset():
    """生成管理器模型数据集"""
    data = []
    
    # 对话管理相关的动作和资源
    dialogue_actions = [
        "create_session", "update_session", "close_session", "switch_model", 
        "allocate_resource", "track_context", "manage_dialogue_flow", "handle_user_intent", 
        "generate_response_plan", "coordinate_models", "maintain_session_state", 
        "handle_interruptions", "process_feedback", "manage_memory", "update_context"
    ]
    dialogue_resources = [
        "conversation", "context", "language_model", "response_generator", 
        "memory", "intent_classifier", "dialogue_state_tracker", "knowledge_base",
        "user_profile", "session_history", "feedback_processor"
    ]
    
    for i in range(5000):  # 增加数据量到5000条
        action = random.choice(dialogue_actions)
        resource = random.choice(dialogue_resources)
        
        # 根据动作和资源设置更相关的元数据
        metadata = {
            "user_id": f"user_{random.randint(1, 500)}",
            "session_id": f"session_{random.randint(1, 1000)}",
            "duration": random.uniform(0.1, 60.0),
            "priority": random.choice(["low", "medium", "high"]),
            "user_intent": random.choice(["inform", "request", "question", "command", "feedback"])
        }
        
        # 对话特定的元数据
        if action in ["create_session", "update_session", "close_session"]:
            metadata["conversation_length"] = random.randint(1, 30)  # 对话轮数
            metadata["active_models"] = random.sample(
                ["language", "manager", "knowledge", "emotion", "vision_image"], 
                k=random.randint(1, 4)
            )
            metadata["session_type"] = random.choice(["general", "specific_task", "chat", "information_retrieval"])
        elif action == "switch_model":
            metadata["from_model"] = random.choice(["language", "manager", "knowledge", "emotion", "vision_image"])
            metadata["to_model"] = random.choice(["language", "manager", "knowledge", "emotion", "vision_image"])
            metadata["reason"] = random.choice(["task_type", "load_balance", "performance", "user_preference"])
            metadata["switch_time"] = random.uniform(0.1, 2.0)
        elif action == "track_context" or action == "update_context":
            metadata["context_size"] = random.randint(1, 200)  # 上下文大小
            metadata["retention_level"] = random.choice(["short", "medium", "long"])
            metadata["context_relevance"] = random.uniform(0.0, 1.0)
        elif action == "allocate_resource":
            metadata["resource_type"] = random.choice(["memory", "cpu", "gpu", "disk"])
            metadata["allocation_amount"] = random.uniform(0.1, 1.0)
            metadata["resource_usage_before"] = random.uniform(0.0, 0.9)
            metadata["resource_usage_after"] = random.uniform(0.1, 1.0)
        elif action == "manage_dialogue_flow":
            metadata["current_turn"] = random.randint(1, 30)
            metadata["dialogue_state"] = random.choice(["start", "in_progress", "ending", "completed"])
            metadata["expected_next_action"] = random.choice(dialogue_actions)
        elif action == "handle_user_intent":
            metadata["intent_confidence"] = random.uniform(0.5, 1.0)
            metadata["intent_category"] = random.choice(["general", "specific", "complex"])
            metadata["required_resources"] = random.sample(dialogue_resources, k=random.randint(1, 3))
        elif action == "generate_response_plan":
            metadata["response_type"] = random.choice(["informative", "question", "command", "confirmation"])
            metadata["plan_complexity"] = random.choice(["simple", "moderate", "complex"])
            metadata["required_models"] = random.sample(
                ["language", "manager", "knowledge", "emotion"], 
                k=random.randint(1, 3)
            )
        elif action == "coordinate_models":
            metadata["model_count"] = random.randint(1, 5)
            metadata["coordination_strategy"] = random.choice(["sequential", "parallel", "hybrid"])
            metadata["communication_overhead"] = random.uniform(0.0, 0.5)
        elif action == "maintain_session_state":
            metadata["state_size"] = random.randint(10, 500)
            metadata["state_consistency"] = random.uniform(0.8, 1.0)
            metadata["last_updated"] = int(time.time()) - random.randint(0, 3600)
        elif action == "handle_interruptions":
            metadata["interruption_type"] = random.choice(["external", "internal", "user_initiated"])
            metadata["recovery_time"] = random.uniform(0.1, 5.0)
            metadata["success_rate"] = random.uniform(0.7, 1.0)
        elif action == "process_feedback":
            metadata["feedback_type"] = random.choice(["positive", "negative", "neutral"])
            metadata["feedback_intensity"] = random.uniform(0.0, 1.0)
            metadata["action_taken"] = random.choice(["none", "adjust_response", "apologize", "provide_alternative"])
        elif action == "manage_memory":
            metadata["memory_type"] = random.choice(["short_term", "long_term", "episodic"])
            metadata["memory_usage"] = random.uniform(0.0, 1.0)
            metadata["cleanup_performed"] = random.choice([True, False])
        
        data.append({
            "id": f"manager_data_{i}",
            "timestamp": int(time.time()) - i * 1800,  # 更频繁的时间戳
            "action": action,
            "resource": resource,
            "status": random.choice(["success", "failed", "pending"]),
            "features": [random.random() for _ in range(12)],  # 增加特征数量
            "metadata": metadata
        })
    
    return data

def generate_language_dataset():
    """生成语言模型数据集"""
    data = []
    
    # 对话主题和样本
    dialogue_topics = [
        {"topic": "科技", "dialogues": [
            "用户：你好，能给我介绍一下人工智能吗？\n助手：当然可以！人工智能是研究使计算机能够模拟人类智能行为的技术，包括机器学习、自然语言处理等领域。",
            "用户：人工智能有哪些应用？\n助手：人工智能应用非常广泛，比如智能助手、自动驾驶、医疗诊断、推荐系统等。",
            "用户：机器学习是什么？\n助手：机器学习是人工智能的一个分支，通过算法使计算机从数据中学习并做出预测或决策。",
            "用户：你好，我想了解一下人工智能\n助手：好的，人工智能是指让计算机模拟人类智能的技术\n用户：那它能做什么呢？\n助手：人工智能可以做很多事情，比如语音识别、图像识别、自然语言处理等\n用户：真的吗？那太厉害了\n助手：是的，人工智能正在改变我们的生活方式",
            "用户：什么是深度学习？\n助手：深度学习是机器学习的一个分支，它使用多层神经网络来模拟人类的大脑\n用户：它和传统机器学习有什么不同？\n助手：深度学习可以自动学习特征，而传统机器学习需要手动提取特征\n用户：哦，原来是这样\n助手：是的，深度学习在图像识别和自然语言处理等领域取得了很大的突破"
        ]},
        {"topic": "历史", "dialogues": [
            "用户：你知道秦始皇统一六国的时间吗？\n助手：秦始皇在公元前221年统一了六国，建立了中国历史上第一个统一的中央集权国家。",
            "用户：兵马俑是什么时候发现的？\n助手：兵马俑是1974年在陕西省西安市临潼区被农民打井时发现的。",
            "用户：唐朝的盛世是什么时候？\n助手：唐朝的盛世主要包括贞观之治（唐太宗时期）和开元盛世（唐玄宗前期）。",
            "用户：你好，能给我讲讲中国古代的四大发明吗？\n助手：当然可以，中国古代的四大发明是造纸术、印刷术、火药和指南针\n用户：它们分别是谁发明的呢？\n助手：造纸术是蔡伦改进的，印刷术是毕昇发明的，火药和指南针的发明者已经不可考了\n用户：这些发明对世界有什么影响？\n助手：这些发明对世界文明的发展产生了深远的影响，促进了文化的传播和科技的进步",
            "用户：你知道三国时期的主要人物吗？\n助手：当然知道，三国时期的主要人物有曹操、刘备、孙权、诸葛亮、关羽、张飞等\n用户：他们之间发生了哪些重要的战役？\n助手：三国时期的重要战役有赤壁之战、官渡之战、夷陵之战等\n用户：赤壁之战是谁赢了？\n助手：赤壁之战是孙权和刘备的联军打败了曹操的军队"
        ]},
        {"topic": "文化", "dialogues": [
            "用户：中国的四大发明是什么？\n助手：中国的四大发明是造纸术、印刷术、火药和指南针。",
            "用户：春节有哪些传统习俗？\n助手：春节的传统习俗包括贴春联、吃年夜饭、放鞭炮、拜年、发红包等。",
            "用户：京剧的角色分为哪几类？\n助手：京剧的角色主要分为生、旦、净、丑四大行当。",
            "用户：你好，我想了解一下中国的传统节日\n助手：好的，中国有很多传统节日，比如春节、中秋节、端午节、清明节等\n用户：春节是怎么来的？\n助手：春节起源于殷商时期年头岁尾的祭神祭祖活动\n用户：春节有哪些传统习俗？\n助手：春节的传统习俗有贴春联、吃年夜饭、放鞭炮、拜年、发红包等",
            "用户：你知道中国的书法吗？\n助手：当然知道，中国书法是中国特有的一种传统艺术形式\n用户：书法有哪些字体？\n助手：书法主要有楷书、行书、草书、隶书、篆书等字体\n用户：谁是中国古代著名的书法家？\n助手：中国古代著名的书法家有王羲之、颜真卿、柳公权、欧阳询等"
        ]},
        {"topic": "生活", "dialogues": [
            "用户：今天天气怎么样？\n助手：今天天气晴朗，温度在20-25度之间，非常适合外出。",
            "用户：能推荐一些好吃的餐厅吗？\n助手：当然可以！我推荐市中心的川菜馆和郊区的农家菜，味道都很不错。",
            "用户：怎么提高睡眠质量？\n助手：提高睡眠质量可以试试保持规律作息、睡前避免电子设备、营造舒适的睡眠环境等方法。",
            "用户：你好，我最近失眠了，有什么好的建议吗？\n助手：失眠可能是由压力、焦虑或生活习惯引起的，建议你保持规律的作息时间，睡前避免使用电子设备，喝一杯温牛奶，或者尝试冥想\n用户：我每天都很晚才睡，早上又起不来\n助手：建议你逐渐调整作息时间，每天提前15分钟睡觉，早上提前15分钟起床，这样更容易适应\n用户：好的，我会尝试的，谢谢你\n助手：不客气，祝你早日改善睡眠质量",
            "用户：你好，能推荐一些健康的饮食习惯吗？\n助手：当然可以，健康的饮食习惯包括多吃蔬菜水果，少吃油腻和辛辣的食物，保持饮食均衡，定时定量进食\n用户：我平时喜欢吃快餐，这样对健康有影响吗？\n助手：经常吃快餐可能会导致营养不均衡，增加肥胖和心血管疾病的风险，建议你尽量少吃\n用户：那我应该怎么调整呢？\n助手：你可以尝试自己做饭，选择新鲜的食材，控制油盐的摄入量，多吃一些粗粮"
        ]},
        {"topic": "学习", "dialogues": [
            "用户：怎么学习Python编程？\n助手：学习Python可以从基础语法开始，多做练习项目，还可以参考网上的教程和文档。",
            "用户：如何提高记忆力？\n助手：提高记忆力可以使用重复记忆、联想记忆、间隔复习等方法，同时保持良好的作息和饮食。",
            "用户：学习英语的好方法有哪些？\n助手：学习英语的好方法包括多听多说、阅读英文材料、使用单词记忆软件、参加语言交换等。",
            "用户：你好，我想学习编程，应该从哪里开始？\n助手：建议你从Python开始学习，因为Python语法简单，容易上手，适合初学者\n用户：学习编程需要买什么书吗？\n助手：可以买一些入门级的编程书籍，比如《Python编程：从入门到实践》，同时也可以参考网上的免费教程\n用户：学习编程需要每天练习吗？\n助手：是的，编程需要大量的练习，建议你每天至少练习1-2小时，这样才能提高编程技能",
            "用户：你好，我在学习英语时遇到了困难，记不住单词\n助手：记单词可以使用一些有效的方法，比如联想记忆、语境记忆、间隔复习等\n用户：什么是间隔复习？\n助手：间隔复习是指在学习新单词后，按照一定的时间间隔进行复习，这样可以提高记忆效果\n用户：还有其他建议吗？\n助手：你可以尝试在日常生活中使用英语，比如看英文电影、听英文歌曲、和外国人交流等"
        ]},
        {"topic": "旅游", "dialogues": [
            "用户：你好，我想去北京旅游，有什么推荐的景点吗？\n助手：北京有很多著名的景点，比如故宫、长城、天安门广场、颐和园、天坛等\n用户：故宫需要提前预约吗？\n助手：是的，故宫需要提前在网上预约门票，建议你提前一周预约\n用户：北京的天气怎么样？\n助手：北京的气候四季分明，春季多风，夏季炎热，秋季凉爽，冬季寒冷\n用户：我应该带什么衣服？\n助手：这取决于你什么时候去，如果是夏季，建议带轻薄的衣服，做好防晒措施；如果是冬季，建议带厚外套、帽子、手套等保暖物品",
            "用户：你好，我想去上海旅游\n助手：上海是一个现代化的大都市，有很多值得一去的地方，比如外滩、东方明珠、豫园、南京路等\n用户：外滩有什么好玩的？\n助手：外滩可以欣赏到黄浦江两岸的美景，尤其是晚上的夜景非常漂亮\n用户：上海的美食有哪些？\n助手：上海的美食有小笼包、生煎包、南翔馒头、本帮菜等"
        ]}
    ]
    
    # 生成对话数据集
    id_counter = 0
    for topic_data in dialogue_topics:
        for dialogue in topic_data["dialogues"]:
            # 生成多个变体
            for _ in range(30):  # 每个对话样本生成30个变体
                data.append({
                    "id": f"lang_data_{id_counter}",
                    "text": dialogue,
                    "timestamp": int(time.time()) - id_counter * 3600,
                    "source": "conversation",
                    "features": [random.random() for _ in range(10)],
                    "metadata": {
                        "language": "zh-CN",
                        "length": len(dialogue),
                        "sentiment": random.uniform(-1.0, 1.0),
                        "topic": topic_data["topic"],
                        "dialogue_turns": dialogue.count("\n") + 1
                    }
                })
                id_counter += 1
    
    # 确保至少有20000条数据
    while len(data) < 20000:
        topic_data = random.choice(dialogue_topics)
        dialogue = random.choice(topic_data["dialogues"])
        data.append({
            "id": f"lang_data_{len(data)}",
            "text": dialogue,
            "timestamp": int(time.time()) - len(data) * 3600,
            "source": "conversation",
            "features": [random.random() for _ in range(10)],
            "metadata": {
                "language": "zh-CN",
                "length": len(dialogue),
                "sentiment": random.uniform(-1.0, 1.0),
                "topic": topic_data["topic"],
                "dialogue_turns": dialogue.count("\n") + 1
            }
        })
    
    return data

def generate_audio_dataset():
    """生成音频模型数据集"""
    data = []
    audio_types = ["speech", "music", "environmental", "animal", "machine"]
    for i in range(2000):
        data.append({
            "id": f"audio_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "audio_type": random.choice(audio_types),
            "duration": random.uniform(0.5, 60.0),
            "sample_rate": random.choice([8000, 16000, 22050, 44100, 48000]),
            "channels": random.choice([1, 2]),
            "features": [random.random() for _ in range(12)],
            "metadata": {
                "signal_strength": random.uniform(0.5, 1.0),
                "noise_level": random.uniform(0.0, 0.5),
                "source": random.choice(["microphone", "recording", "stream"])
            }
        })
    return data

def generate_vision_image_dataset():
    """生成视觉图像模型数据集"""
    data = []
    image_types = ["natural", "artificial", "animal", "plant", "human", "object", "scene"]
    for i in range(5000):
        data.append({
            "id": f"image_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "image_type": random.choice(image_types),
            "width": random.randint(100, 1920),
            "height": random.randint(100, 1080),
            "color_channels": random.choice([1, 3]),
            "features": [random.random() for _ in range(16)],
            "metadata": {
                "brightness": random.uniform(0.0, 1.0),
                "contrast": random.uniform(0.0, 1.0),
                "sharpness": random.uniform(0.0, 1.0)
            }
        })
    return data

def generate_vision_video_dataset():
    """生成视觉视频模型数据集"""
    data = []
    video_types = ["movie", "tv_show", "documentary", "news", "sports", "music_video", "user_generated"]
    for i in range(1000):
        data.append({
            "id": f"video_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "video_type": random.choice(video_types),
            "duration": random.uniform(10.0, 3600.0),
            "resolution": random.choice(["480p", "720p", "1080p", "2K", "4K"]),
            "frame_rate": random.choice([24, 25, 30, 60]),
            "features": [random.random() for _ in range(12)],
            "metadata": {
                "codec": random.choice(["H.264", "H.265", "VP9", "AV1"]),
                "bitrate": random.uniform(1, 50),
                "source": random.choice(["streaming", "file", "recording"])
            }
        })
    return data

def generate_spatial_dataset():
    """生成空间模型数据集"""
    data = []
    spatial_types = ["indoor", "outdoor", "urban", "natural", "industrial"]
    for i in range(3000):
        data.append({
            "id": f"spatial_data_{i}",
            "location": f"{random.choice(spatial_types)}_location_{i}",
            "category": random.choice(spatial_types),
            "coordinates": {
                "x": random.uniform(-1000, 1000),
                "y": random.uniform(-1000, 1000),
                "z": random.uniform(0, 500)
            },
            "features": [random.random() for _ in range(10)],
            "metadata": {
                "temperature": random.uniform(-20, 50),
                "humidity": random.uniform(0, 100),
                "lighting": random.uniform(0, 1000)
            }
        })
    return data

def generate_sensor_dataset():
    """生成传感器模型数据集"""
    data = []
    sensor_types = ["temperature", "humidity", "pressure", "light", "sound", "motion", "proximity"]
    environments = ["indoor", "outdoor", "industrial", "domestic", "automotive"]
    for i in range(5000):
        data.append({
            "sensor_id": f"sensor_{i}",
            "sensor_type": random.choice(sensor_types),
            "environment": random.choice(environments),
            "timestamp": int(time.time()) - i * 60,
            "value": random.uniform(0, 100),
            "features": [random.random() for _ in range(8)],
            "metadata": {
                "battery_level": random.uniform(0, 100),
                "signal_strength": random.uniform(-100, 0),
                "calibration_status": random.choice(["calibrated", "needs_calibration"])
            }
        })
    return data

def generate_computer_dataset():
    """生成计算机模型数据集"""
    data = []
    computer_types = ["cpu", "memory", "disk", "network", "process"]
    for i in range(5000):
        data.append({
            "id": f"computer_data_{i}",
            "timestamp": int(time.time()) - i * 60,
            "component_type": random.choice(computer_types),
            "usage": random.uniform(0, 100),
            "temperature": random.uniform(20, 80),
            "features": [random.random() for _ in range(8)],
            "metadata": {
                "status": random.choice(["normal", "warning", "error"]),
                "model": f"model_{random.randint(1, 100)}",
                "manufacturer": random.choice(["Intel", "AMD", "NVIDIA", "Samsung", "Western Digital"])
            }
        })
    return data

def generate_motion_dataset():
    """生成运动模型数据集"""
    data = []
    motion_types = ["walking", "running", "jumping", "dancing", "swimming", "cycling"]
    for i in range(3000):
        data.append({
            "id": f"motion_data_{i}",
            "timestamp": int(time.time()) - i * 60,
            "motion_type": random.choice(motion_types),
            "speed": random.uniform(0, 30),
            "acceleration": random.uniform(-5, 5),
            "direction": random.uniform(0, 360),
            "features": [random.random() for _ in range(10)],
            "metadata": {
                "subject_id": f"subject_{random.randint(1, 50)}",
                "sensor_count": random.randint(1, 10),
                "confidence": random.uniform(0.5, 1.0)
            }
        })
    return data

def generate_knowledge_dataset():
    """生成知识库模型数据集"""
    data = []
    knowledge_types = ["fact", "concept", "theory", "rule", "principle"]
    domains = ["science", "technology", "history", "philosophy", "mathematics"]
    for i in range(10000):
        data.append({
            "id": f"knowledge_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "knowledge_type": random.choice(knowledge_types),
            "domain": random.choice(domains),
            "content": f"这是一个关于{random.choice(domains)}的{random.choice(knowledge_types)}知识内容。" * random.randint(1, 5),
            "features": [random.random() for _ in range(12)],
            "metadata": {
                "source": random.choice(["book", "article", "website", "expert"]),
                "confidence": random.uniform(0.5, 1.0),
                "tags": random.sample(domains, k=random.randint(1, 3))
            }
        })
    return data

def generate_programming_dataset():
    """生成编程模型数据集"""
    data = []
    languages = ["python", "javascript", "java", "cpp", "c", "go", "rust"]
    for i in range(8000):
        data.append({
            "id": f"programming_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "language": random.choice(languages),
            "code": f"print('Hello, World!')\nfor i in range(10):\n    print(i)" * random.randint(1, 5),
            "features": [random.random() for _ in range(10)],
            "metadata": {
                "line_count": random.randint(1, 100),
                "complexity": random.uniform(0, 10),
                "style": random.choice(["functional", "object_oriented", "procedural"])
            }
        })
    return data

def generate_planning_dataset():
    """生成规划模型数据集"""
    data = []
    plan_types = ["project", "trip", "meeting", "production", "research"]
    for i in range(2000):
        data.append({
            "id": f"planning_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "plan_type": random.choice(plan_types),
            "duration": random.randint(1, 365),
            "tasks": random.randint(1, 50),
            "resources": random.randint(1, 20),
            "features": [random.random() for _ in range(10)],
            "metadata": {
                "status": random.choice(["planning", "executing", "completed"]),
                "priority": random.choice(["low", "medium", "high"]),
                "risk": random.uniform(0, 100)
            }
        })
    return data

def generate_autonomous_dataset():
    """生成自主模型数据集"""
    data = []
    autonomous_types = ["robot", "vehicle", "drone", "system", "agent"]
    for i in range(3000):
        data.append({
            "id": f"autonomous_data_{i}",
            "timestamp": int(time.time()) - i * 60,
            "agent_type": random.choice(autonomous_types),
            "action": random.choice(["move", "observe", "communicate", "decide", "execute"]),
            "environment": random.choice(["indoor", "outdoor", "virtual"]),
            "features": [random.random() for _ in range(12)],
            "metadata": {
                "battery": random.uniform(0, 100),
                "signal": random.uniform(0, 100),
                "status": random.choice(["normal", "warning", "error"])
            }
        })
    return data

def generate_emotion_dataset():
    """生成情感模型数据集"""
    data = []
    emotions = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "neutral"]
    for i in range(5000):
        data.append({
            "id": f"emotion_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "emotion": random.choice(emotions),
            "intensity": random.uniform(0, 100),
            "text": f"这是一个表达{random.choice(emotions)}情感的文本。" * random.randint(1, 5),
            "features": [random.random() for _ in range(8)],
            "metadata": {
                "source": random.choice(["text", "audio", "video"]),
                "confidence": random.uniform(0.5, 1.0),
                "context": random.choice(["conversation", "social_media", "article"])
            }
        })
    return data

def generate_prediction_dataset():
    """生成预测模型数据集"""
    data = []
    prediction_types = ["weather", "stock", "sales", "traffic", "demand"]
    for i in range(10000):
        data.append({
            "id": f"prediction_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "prediction_type": random.choice(prediction_types),
            "actual_value": random.uniform(0, 1000),
            "predicted_value": random.uniform(0, 1000),
            "features": [random.random() for _ in range(12)],
            "metadata": {
                "accuracy": random.uniform(0.5, 1.0),
                "confidence": random.uniform(0.5, 1.0),
                "horizon": random.randint(1, 365)
            }
        })
    return data

def generate_collaboration_dataset():
    """生成协作模型数据集"""
    data = []
    collaboration_types = ["team", "project", "meeting", "document", "code"]
    for i in range(3000):
        data.append({
            "id": f"collaboration_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "collaboration_type": random.choice(collaboration_types),
            "participants": random.randint(2, 50),
            "activity": random.choice(["create", "edit", "comment", "review", "approve"]),
            "features": [random.random() for _ in range(10)],
            "metadata": {
                "status": random.choice(["active", "completed", "archived"]),
                "platform": random.choice(["github", "slack", "microsoft_teams", "google_workspace"]),
                "productivity": random.uniform(0, 100)
            }
        })
    return data

def generate_optimization_dataset():
    """生成优化模型数据集"""
    data = []
    optimization_types = ["resource", "schedule", "cost", "performance", "quality"]
    for i in range(5000):
        data.append({
            "id": f"optimization_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "optimization_type": random.choice(optimization_types),
            "current_value": random.uniform(0, 100),
            "optimized_value": random.uniform(0, 100),
            "features": [random.random() for _ in range(10)],
            "metadata": {
                "improvement": random.uniform(-10, 50),
                "constraints": random.randint(1, 10),
                "algorithm": random.choice(["genetic", "gradient_descent", "simulated_annealing", "particle_swarm"])
            }
        })
    return data

def generate_finance_dataset():
    """生成金融模型数据集"""
    data = []
    finance_types = ["stock", "bond", "currency", "commodity", "real_estate"]
    for i in range(10000):
        data.append({
            "id": f"finance_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "asset_type": random.choice(finance_types),
            "price": random.uniform(1, 10000),
            "volume": random.uniform(1000, 1000000),
            "features": [random.random() for _ in range(12)],
            "metadata": {
                "symbol": f"SYMBOL_{i}",
                "exchange": random.choice(["NYSE", "NASDAQ", "LSE", "Tokyo", "Shanghai"]),
                "change": random.uniform(-10, 10)
            }
        })
    return data

def generate_medical_dataset():
    """生成医疗模型数据集"""
    data = []
    medical_types = ["diagnosis", "treatment", "patient", "symptom", "medication"]
    for i in range(5000):
        data.append({
            "id": f"medical_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "medical_type": random.choice(medical_types),
            "patient_id": f"patient_{i}",
            "age": random.randint(0, 100),
            "features": [random.random() for _ in range(15)],
            "metadata": {
                "gender": random.choice(["male", "female", "other"]),
                "department": random.choice(["cardiology", "neurology", "orthopedics", "pediatrics", "psychiatry"]),
                "severity": random.choice(["mild", "moderate", "severe"])
            }
        })
    return data

def generate_value_alignment_dataset():
    """生成价值对齐模型数据集"""
    data = []
    values = ["fairness", "justice", "equality", "freedom", "safety", "privacy"]
    for i in range(3000):
        data.append({
            "id": f"value_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "value": random.choice(values),
            "action": f"这是一个与{random.choice(values)}相关的行动描述。" * random.randint(1, 5),
            "alignment_score": random.uniform(0, 100),
            "features": [random.random() for _ in range(10)],
            "metadata": {
                "context": random.choice(["social", "ethical", "legal", "professional"]),
                "confidence": random.uniform(0.5, 1.0),
                "source": random.choice(["human", "ai", "committee"])
            }
        })
    return data

def generate_stereo_vision_dataset():
    """生成立体视觉模型数据集"""
    data = []
    vision_types = ["depth", "3d_reconstruction", "object_recognition", "scene_understanding"]
    for i in range(3000):
        data.append({
            "id": f"stereo_data_{i}",
            "timestamp": int(time.time()) - i * 3600,
            "vision_type": random.choice(vision_types),
            "resolution": random.choice(["480p", "720p", "1080p", "2K", "4K"]),
            "baseline": random.uniform(0.01, 0.5),
            "features": [random.random() for _ in range(12)],
            "metadata": {
                "depth_accuracy": random.uniform(0.5, 1.0),
                "camera_distance": random.uniform(0.1, 100),
                "lighting": random.uniform(0, 1000)
            }
        })
    return data

def generate_default_dataset():
    """生成默认模型数据集"""
    data = []
    for i in range(100):
        data.append({
            "id": f"default_data_{i}",
            "features": [random.random() for _ in range(10)],
            "label": random.choice([0, 1]),
            "timestamp": int(time.time()) - i * 3600,
            "metadata": {
                "source": "generated",
                "quality": random.uniform(0.5, 1.0)
            }
        })
    return data

if __name__ == "__main__":
    generate_datasets()
