#!/usr/bin/env python3
"""
生成训练数据脚本 - 为所有模型生成适合的训练数据集
Generate Training Data Script - Generate suitable training datasets for all models

根据training_plan.md中的数据集要求，为每个模型生成或准备相应的训练数据
"""

import os
import sys
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

# 创建确定性随机实例
random = DeterministicRandom(seed=42)
import time
import numpy as np
import sympy
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.model_registry import ModelRegistry
from core.training_preparation import TrainingPreparation, create_training_preparation
from core.training_manager import TrainingManager

def create_synthetic_text_data(size: int, data_type: str = "general") -> List[str]:
    """生成合成文本数据
    Generate synthetic text data
    
    Args:
        size: 数据规模
        data_type: 数据类型 (general, knowledge, language, etc.)
        
    Returns:
        List[str]: 文本数据列表
    """
    data = []
    
    # 通用文本模板
    general_templates = [
        "这是一个关于{}的示例文本。",
        "{}是一种重要的概念，涉及{}等方面。",
        "在{}领域中，{}被广泛应用于{}。",
        "研究表明，{}与{}之间存在着密切的关系。",
        "随着{}的发展，{}的应用前景越来越广阔。"
    ]
    
    # 知识模型文本模板
    knowledge_templates = [
        "{}是{}的重要组成部分，具有{}等特点。",
        "根据百科全书，{}的定义是{}。",
        "在学术研究中，{}被认为是{}的基础。",
        "历史上，{}的发展经历了{}等阶段。",
        "{}的主要特征包括{}。",
        "{} is an important part of {}, which has characteristics such as {}.",
        "According to the encyclopedia, the definition of {} is {}.",
        "In academic research, {} is considered the foundation of {}."
    ]
    
    # 语言模型文本模板
    language_templates = [
        "用户问：'{}'，回答应该是：'{}'。",
        "{}是一个常见的表达方式，通常用于{}场合。",
        "句子'{}'的意思是{}。",
        "在{}语境下，{}可以理解为{}。",
        "{}的同义词包括{}等。",
        "User asks: '{}', the answer should be: '{}'.",
        "'{}' is a common expression used in {} situations.",
        "What is the sum of {} and {}? The answer is {}.",
        "Calculate {} multiplied by {}, the result is {}.",
        "What is {} minus {}? It equals {}."
    ]
    
    # 根据数据类型选择模板
    if data_type == "knowledge":
        templates = knowledge_templates
        subjects = ["物理学", "化学", "生物学", "数学", "计算机科学", "历史", "地理", "经济学", "Physics", "Chemistry", "Biology", "Mathematics", "Computer Science", "History"]
        details = ["基本原理", "核心概念", "重要理论", "关键发现", "实际应用", "basic principles", "core concepts", "important theories", "key findings", "practical applications"]
    elif data_type == "language":
        templates = language_templates
        
        # 中文问题和答案
        cn_questions = ["如何学习编程？", "今天天气怎么样？", "什么是人工智能？", "如何保持健康？", "什么是区块链？"]
        cn_answers = ["可以通过在线课程和实践项目学习编程。", "今天天气晴朗，温度适中。", "人工智能是模拟人类智能的技术。", "保持健康需要均衡饮食和适量运动。", "区块链是一种分布式账本技术。"]
        
        # 英文问题和答案
        en_questions = ["How are you?", "What's your name?", "What time is it?", "Where are you from?", "How do I get to the park?"]
        en_answers = ["I'm doing well, thank you!", "My name is AI Assistant.", "It's 12:30 PM.", "I'm from the cloud.", "You can take the bus or walk there."]
        
        # 数学问题和答案（将在生成时动态计算）
        
        contexts = ["日常对话", "学术讨论", "技术交流", "生活咨询", "专业解释", "daily conversation", "academic discussion", "technical communication"]
        meanings = ["询问学习方法", "询问天气状况", "询问概念定义", "询问健康建议", "询问技术原理", "asking about well-being", "asking for time", "asking for directions"]
        
        # 合并所有问题和答案
        questions = cn_questions + en_questions
        answers = cn_answers + en_answers
    else:
        templates = general_templates
        subjects = ["科技", "教育", "健康", "环境", "经济", "文化", "体育", "艺术", "Technology", "Education", "Health", "Environment"]
        details = ["理论研究", "实际应用", "发展趋势", "社会影响", "未来展望", "theoretical research", "practical applications", "development trends"]
    
    # 生成数据
    for i in range(size):
        template = templates[hash(str(templates) + str(i)) % len(templates)]
        if data_type == "knowledge":
            text = template.format(
                subjects[hash(str(subjects) + str(i) + "0") % len(subjects)],
                subjects[hash(str(subjects) + str(i) + "1") % len(subjects)],
                details[hash(str(details) + str(i)) % len(details)]
            )
        elif data_type == "language":
            if "用户问" in template:
                # 中文对话模板
                text = template.format(
                    cn_questions[hash(str(cn_questions) + str(i)) % len(cn_questions)],
                    cn_answers[hash(str(cn_answers) + str(i)) % len(cn_answers)]
                )
            elif "User asks" in template:
                # 英文对话模板
                text = template.format(
                    en_questions[hash(str(en_questions) + str(i)) % len(en_questions)],
                    en_answers[hash(str(en_answers) + str(i)) % len(en_answers)]
                )
            elif "sum" in template or "plus" in template:
                # 加法计算模板
                a = 1 + (hash(str(i) + "a") % 100)
                b = 1 + (hash(str(i) + "b") % 100)
                c = a + b
                text = template.format(a, b, c)
            elif "multiplied by" in template or "times" in template:
                # 乘法计算模板
                a = 1 + (hash(str(i) + "mul_a") % 20)
                b = 1 + (hash(str(i) + "mul_b") % 20)
                c = a * b
                text = template.format(a, b, c)
            elif "minus" in template:
                # 减法计算模板
                a = 1 + (hash(str(i) + "sub_a") % 100)
                b = 1 + (hash(str(i) + "sub_b") % a)  # 确保结果为正数
                c = a - b
                text = template.format(a, b, c)
            elif "在" in template and "语境下" in template:
                
                text = template.format(
                    contexts[hash(str(contexts) + str(i) + "0") % len(contexts)],
                    questions[hash(str(questions) + str(i) + "1") % len(questions)],
                    meanings[hash(str(meanings) + str(i) + "2") % len(meanings)]
                )
            else:
                # 其他模板
                text = template.format(
                    questions[hash(str(questions) + str(i) + "3") % len(questions)],
                    meanings[hash(str(meanings) + str(i) + "4") % len(meanings)]
                )
        else:
            text = template.format(
                subjects[hash(str(subjects) + str(i) + "5") % len(subjects)],
                details[hash(str(details) + str(i) + "6") % len(details)]
            )
        data.append(text)
    
    return data

def create_synthetic_image_data(size: int) -> List[Dict[str, Any]]:
    """生成合成图像数据（模拟图像路径和标签）
    Generate synthetic image data (simulate image paths and labels)
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 图像数据列表，包含图像路径和标签
    """
    data = []
    categories = ["动物", "植物", "建筑", "交通工具", "食物", "电子产品", "自然风景", "人造物品"]
    
    for i in range(size):
        image_data = {
            "image_path": f"/data/images/image_{i}.jpg",
            "label": random.choice(categories),
            "width": random.randint(256, 1024),
            "height": random.randint(256, 1024),
            "source": "synthetic"
        }
        data.append(image_data)
    
    return data

def create_synthetic_audio_data(size: int) -> List[Dict[str, Any]]:
    """生成合成音频数据（模拟音频路径和标签）
    Generate synthetic audio data (simulate audio paths and labels)
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 音频数据列表，包含音频路径和标签
    """
    data = []
    categories = ["语音", "音乐", "环境声音", "动物叫声", "机械声音", "交通工具声音"]
    
    for i in range(size):
        audio_data = {
            "audio_path": f"/data/audio/audio_{i}.wav",
            "label": random.choice(categories),
            "duration": random.uniform(1.0, 60.0),
            "sample_rate": 44100,
            "source": "synthetic"
        }
        data.append(audio_data)
    
    return data

def create_synthetic_video_data(size: int) -> List[Dict[str, Any]]:
    """生成合成视频数据（模拟视频路径和标签）
    Generate synthetic video data (simulate video paths and labels)
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 视频数据列表，包含视频路径和标签
    """
    data = []
    categories = ["电影片段", "新闻报道", "教育视频", "体育比赛", "动物行为", "自然景观", "科技演示"]
    
    for i in range(size):
        video_data = {
            "video_path": f"/data/videos/video_{i}.mp4",
            "label": random.choice(categories),
            "duration": random.uniform(5.0, 300.0),
            "resolution": "1920x1080",
            "source": "synthetic"
        }
        data.append(video_data)
    
    return data

def create_synthetic_spatial_data(size: int) -> List[Dict[str, Any]]:
    """生成合成空间数据
    Generate synthetic spatial data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 空间数据列表
    """
    data = []
    
    for i in range(size):
        spatial_data = {
            "scene_id": f"scene_{i}",
            "3d_coordinates": {
                "x": random.uniform(-10.0, 10.0),
                "y": random.uniform(-10.0, 10.0),
                "z": random.uniform(-10.0, 10.0)
            },
            "objects": [
                {
                    "name": f"object_{j}_{i}",
                    "type": random.choice(["家具", "电器", "装饰品", "工具", "其他"]),
                    "position": {
                        "x": random.uniform(-5.0, 5.0),
                        "y": random.uniform(-5.0, 5.0),
                        "z": random.uniform(-5.0, 5.0)
                    }
                } for j in range(random.randint(1, 5))
            ],
            "source": "synthetic"
        }
        data.append(spatial_data)
    
    return data

def create_synthetic_sensor_data(size: int) -> List[Dict[str, Any]]:
    """生成合成传感器数据
    Generate synthetic sensor data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 传感器数据列表
    """
    data = []
    sensor_types = ["温度传感器", "湿度传感器", "压力传感器", "加速度传感器", "光线传感器"]
    
    for i in range(size):
        sensor_data = {
            "sensor_id": f"sensor_{i}",
            "sensor_type": random.choice(sensor_types),
            "timestamp": time.time() - random.randint(0, 86400),
            "value": random.uniform(0.0, 100.0),
            "unit": random.choice(["°C", "%", "Pa", "m/s²", "lux"]),
            "source": "synthetic"
        }
        data.append(sensor_data)
    
    return data

def create_synthetic_computer_data(size: int) -> List[Dict[str, Any]]:
    """生成合成计算机命令与响应数据
    Generate synthetic computer command and response data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 计算机数据列表
    """
    data = []
    commands = [
        {"command": "ls", "response": "file1.txt file2.txt dir1"},
        {"command": "pwd", "response": "/home/user"},
        {"command": "mkdir new_dir", "response": ""},
        {"command": "rm file.txt", "response": ""},
        {"command": "cp file1.txt file2.txt", "response": ""},
        {"command": "mv file1.txt new_name.txt", "response": ""},
        {"command": "cat file.txt", "response": "This is the content of file.txt"},
        {"command": "grep 'keyword' file.txt", "response": "Line with keyword"},
        {"command": "find . -name '*.txt'", "response": "./file1.txt ./dir1/file2.txt"},
        {"command": "chmod +x script.sh", "response": ""}
    ]
    
    for i in range(size):
        command_data = random.choice(commands).copy()
        command_data["timestamp"] = time.time() - random.randint(0, 86400)
        command_data["source"] = "synthetic"
        data.append(command_data)
    
    return data

def create_synthetic_motion_data(size: int) -> List[Dict[str, Any]]:
    """生成合成运动数据
    Generate synthetic motion data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 运动数据列表
    """
    data = []
    motion_types = ["行走", "跑步", "跳跃", "攀爬", "旋转", "移动"]
    
    for i in range(size):
        motion_data = {
            "motion_id": f"motion_{i}",
            "motion_type": random.choice(motion_types),
            "start_time": time.time() - random.randint(0, 86400),
            "end_time": time.time() - random.randint(0, 86400),
            "parameters": {
                "speed": random.uniform(0.0, 10.0),
                "distance": random.uniform(0.0, 100.0),
                "direction": random.uniform(0.0, 360.0),
                "acceleration": random.uniform(-2.0, 2.0)
            },
            "source": "synthetic"
        }
        data.append(motion_data)
    
    return data

def create_synthetic_programming_data(size: int) -> List[Dict[str, Any]]:
    """生成合成编程数据
    Generate synthetic programming data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 编程数据列表
    """
    data = []
    languages = ["Python", "Java", "C++", "JavaScript", "Go"]
    problems = [
        "计算两个数的和",
        "判断一个数是否为质数",
        "反转字符串",
        "计算阶乘",
        "查找数组中的最大值",
        "排序数组",
        "计算斐波那契数列",
        "判断回文字符串",
        "计算平均数",
        "统计字符串中的字符出现次数"
    ]
    
    for i in range(size):
        language = random.choice(languages)
        problem = random.choice(problems)
        
        # 生成简单的代码示例
        if problem == "计算两个数的和":
            if language == "Python":
                code = "def add(a, b):\n    return a + b\n\nresult = add(3, 5)\nprint(result)"
            elif language == "Java":
                code = "public class Add {\n    public static void main(String[] args) {\n        int a = 3;\n        int b = 5;\n        int result = a + b;\n        System.out.println(result);\n    }\n}"
            else:
                code = f"// {language} code to add two numbers"
        elif problem == "反转字符串":
            if language == "Python":
                code = "def reverse_string(s):\n    return s[::-1]\n\nresult = reverse_string('hello')\nprint(result)"
            else:
                code = f"// {language} code to reverse a string"
        else:
            code = f"// {language} code to solve: {problem}"
        
        programming_data = {
            "program_id": f"program_{i}",
            "language": language,
            "problem_description": problem,
            "code": code,
            "comments": f"这是一个用{language}语言解决{problem}的示例代码。",
            "source": "synthetic"
        }
        data.append(programming_data)
    
    return data

def create_synthetic_mathematics_data(size: int) -> List[Dict[str, Any]]:
    """生成合成数学数据
    Generate synthetic mathematics data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 数学数据列表，包含问题、解决方案和元数据
    """
    data = []
    
    # 数学问题类型
    problem_types = [
        "algebraic_equation",  # 代数方程
        "calculus_differentiation",  # 微积分求导
        "calculus_integration",  # 微积分积分
        "geometry_area",  # 几何面积
        "geometry_volume",  # 几何体积
        "trigonometry",  # 三角函数
        "statistics",  # 统计
        "linear_algebra"  # 线性代数
    ]
    
    for i in range(size):
        problem_type = random.choice(problem_types)
        
        # 根据问题类型生成具体问题
        if problem_type == "algebraic_equation":
            # 生成线性方程: ax + b = c
            a = random.randint(1, 10)
            b = random.randint(1, 20)
            c = random.randint(b + 1, 50)
            # 解: x = (c - b) / a
            x_solution = (c - b) / a
            problem_text = f"Solve for x: {a}x + {b} = {c}"
            solution_text = f"x = {x_solution}"
            solution_value = x_solution
            
        elif problem_type == "calculus_differentiation":
            # 生成求导问题
            functions = ["x^2", "sin(x)", "cos(x)", "e^x", "ln(x)"]
            func = random.choice(functions)
            problem_text = f"Find the derivative of {func} with respect to x"
            # 使用sympy计算导数
            try:
                x = sympy.symbols('x')
                if func == "x^2":
                    derivative = "2*x"
                elif func == "sin(x)":
                    derivative = "cos(x)"
                elif func == "cos(x)":
                    derivative = "-sin(x)"
                elif func == "e^x":
                    derivative = "e^x"
                elif func == "ln(x)":
                    derivative = "1/x"
                else:
                    derivative = "unknown"
                solution_text = f"Derivative: {derivative}"
                solution_value = derivative
            except Exception:
                solution_text = "Derivative calculation failed"
                solution_value = None
                
        elif problem_type == "calculus_integration":
            # 生成积分问题
            functions = ["2*x", "x^2", "sin(x)", "cos(x)", "e^x"]
            func = random.choice(functions)
            problem_text = f"Find the integral of {func} with respect to x"
            # 使用sympy计算积分
            try:
                x = sympy.symbols('x')
                if func == "2*x":
                    integral = "x^2"
                elif func == "x^2":
                    integral = "x^3/3"
                elif func == "sin(x)":
                    integral = "-cos(x)"
                elif func == "cos(x)":
                    integral = "sin(x)"
                elif func == "e^x":
                    integral = "e^x"
                else:
                    integral = "unknown"
                solution_text = f"Integral: {integral}"
                solution_value = integral
            except Exception:
                solution_text = "Integration calculation failed"
                solution_value = None
                
        elif problem_type == "geometry_area":
            # 几何面积问题
            shapes = ["circle", "rectangle", "triangle", "square"]
            shape = random.choice(shapes)
            if shape == "circle":
                radius = random.uniform(1.0, 10.0)
                area = sympy.pi * radius**2
                problem_text = f"Calculate the area of a circle with radius {radius:.2f}"
                solution_text = f"Area = {area:.4f}"
                solution_value = float(area)
            elif shape == "rectangle":
                length = random.uniform(2.0, 15.0)
                width = random.uniform(2.0, 10.0)
                area = length * width
                problem_text = f"Calculate the area of a rectangle with length {length:.2f} and width {width:.2f}"
                solution_text = f"Area = {area:.4f}"
                solution_value = float(area)
            elif shape == "triangle":
                base = random.uniform(3.0, 12.0)
                height = random.uniform(2.0, 8.0)
                area = 0.5 * base * height
                problem_text = f"Calculate the area of a triangle with base {base:.2f} and height {height:.2f}"
                solution_text = f"Area = {area:.4f}"
                solution_value = float(area)
            else:  # square
                side = random.uniform(2.0, 10.0)
                area = side**2
                problem_text = f"Calculate the area of a square with side length {side:.2f}"
                solution_text = f"Area = {area:.4f}"
                solution_value = float(area)
                
        elif problem_type == "geometry_volume":
            # 几何体积问题
            shapes = ["sphere", "cube", "cylinder", "cone"]
            shape = random.choice(shapes)
            if shape == "sphere":
                radius = random.uniform(1.0, 8.0)
                volume = (4/3) * sympy.pi * radius**3
                problem_text = f"Calculate the volume of a sphere with radius {radius:.2f}"
                solution_text = f"Volume = {volume:.4f}"
                solution_value = float(volume)
            elif shape == "cube":
                side = random.uniform(2.0, 10.0)
                volume = side**3
                problem_text = f"Calculate the volume of a cube with side length {side:.2f}"
                solution_text = f"Volume = {volume:.4f}"
                solution_value = float(volume)
            elif shape == "cylinder":
                radius = random.uniform(1.0, 5.0)
                height = random.uniform(3.0, 12.0)
                volume = sympy.pi * radius**2 * height
                problem_text = f"Calculate the volume of a cylinder with radius {radius:.2f} and height {height:.2f}"
                solution_text = f"Volume = {volume:.4f}"
                solution_value = float(volume)
            else:  # cone
                radius = random.uniform(1.0, 5.0)
                height = random.uniform(3.0, 12.0)
                volume = (1/3) * sympy.pi * radius**2 * height
                problem_text = f"Calculate the volume of a cone with radius {radius:.2f} and height {height:.2f}"
                solution_text = f"Volume = {volume:.4f}"
                solution_value = float(volume)
                
        else:
            # 其他类型使用通用问题
            problem_text = f"Solve this mathematical problem (type: {problem_type})"
            solution_text = f"Solution for {problem_type} problem"
            solution_value = None
        
        # 创建数据条目
        math_data = {
            "math_id": f"math_{i}",
            "problem_type": problem_type,
            "problem_text": problem_text,
            "solution_text": solution_text,
            "solution_value": solution_value,
            "difficulty": random.uniform(0.1, 1.0),
            "domain": random.choice(["algebra", "calculus", "geometry", "trigonometry", "statistics"]),
            "source": "synthetic"
        }
        data.append(math_data)
    
    return data

def create_synthetic_emotion_data(size: int) -> List[Dict[str, Any]]:
    """生成合成情感数据
    Generate synthetic emotion data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 情感数据列表
    """
    data = []
    emotions = ["高兴", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "中立"]
    
    for i in range(size):
        emotion_data = {
            "emotion_id": f"emotion_{i}",
            "text": random.choice([
                "今天天气真好，我很开心！",
                "我考试没通过，感到很沮丧。",
                "有人插队，我非常生气！",
                "晚上独自在家，有点害怕。",
                "收到礼物，我感到很惊讶。",
                "这个食物的味道让我很恶心。",
                "今天和平常一样，没什么特别的。"
            ]),
            "emotion_label": random.choice(emotions),
            "intensity": random.uniform(0.1, 1.0),
            "source": "synthetic"
        }
        data.append(emotion_data)
    
    return data

def create_synthetic_planning_data(size: int) -> List[Dict[str, Any]]:
    """生成合成规划数据
    Generate synthetic planning data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 规划数据列表
    """
    data = []
    tasks = [
        "去超市购物",
        "完成作业",
        "准备晚餐",
        "打扫房间",
        "锻炼身体"
    ]
    
    for i in range(size):
        planning_data = {
            "plan_id": f"plan_{i}",
            "task": random.choice(tasks),
            "steps": [
                {"step": j + 1, "description": f"{random.choice(['首先', '然后', '接着', '之后', '最后'])}做{random.choice(['准备工作', '主要任务', '收尾工作'])}"}
                for j in range(random.randint(2, 5))
            ],
            "estimated_time": random.randint(10, 120),
            "resources": random.choice([["时间", "精力"], ["工具", "材料"], ["金钱", "时间"]]),
            "source": "synthetic"
        }
        data.append(planning_data)
    
    return data

def create_synthetic_prediction_data(size: int) -> List[Dict[str, Any]]:
    """生成合成预测数据
    Generate synthetic prediction data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 预测数据列表
    """
    data = []
    prediction_types = ["股票价格", "天气情况", "销售量", "交通流量", "能源消耗"]
    
    for i in range(size):
        # 生成时间序列数据
        time_series = [random.uniform(50.0, 150.0) for _ in range(10)]
        
        prediction_data = {
            "prediction_id": f"prediction_{i}",
            "prediction_type": random.choice(prediction_types),
            "historical_data": time_series,
            "predicted_value": random.uniform(80.0, 120.0),
            "confidence": random.uniform(0.5, 1.0),
            "timestamp": time.time(),
            "source": "synthetic"
        }
        data.append(prediction_data)
    
    return data

def create_synthetic_collaboration_data(size: int) -> List[Dict[str, Any]]:
    """生成合成协作数据
    Generate synthetic collaboration data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 协作数据列表
    """
    data = []
    roles = ["领导者", "参与者", "协调者", "记录者"]
    
    for i in range(size):
        collaboration_data = {
            "collaboration_id": f"collaboration_{i}",
            "project_name": f"项目_{i}",
            "participants": [
                {
                    "participant_id": f"user_{j}_{i}",
                    "role": random.choice(roles),
                    "contribution": random.uniform(0.1, 1.0)
                } for j in range(random.randint(2, 5))
            ],
            "dialogue": [
                {
                    "speaker": f"user_{j}_{i}",
                    "message": f"关于{random.choice(['任务分配', '时间安排', '资源需求', '问题解决', '进展汇报'])}，我认为应该{random.choice(['优先处理', '重新评估', '寻求帮助', '调整计划', '加快进度'])}"
                } for j in range(random.randint(3, 8))
            ],
            "outcome": random.choice(["任务完成", "计划调整", "问题解决", "需要进一步讨论"]),
            "source": "synthetic"
        }
        data.append(collaboration_data)
    
    return data

def create_synthetic_optimization_data(size: int) -> List[Dict[str, Any]]:
    """生成合成优化数据
    Generate synthetic optimization data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 优化数据列表
    """
    data = []
    optimization_types = ["资源分配", "路径规划", "参数调整", "成本控制", "时间管理"]
    
    for i in range(size):
        optimization_data = {
            "optimization_id": f"optimization_{i}",
            "problem_type": random.choice(optimization_types),
            "objective": random.choice(["最大化收益", "最小化成本", "缩短时间", "提高效率", "优化资源利用"]),
            "variables": {
                f"变量{j}": {
                    "type": random.choice(["连续", "离散", "整数"]),
                    "range": [random.uniform(0.0, 10.0), random.uniform(10.0, 100.0)]
                } for j in range(random.randint(2, 5))
            },
            "constraints": [
                f"{random.choice(['资源限制', '时间限制', '质量要求', '成本限制', '技术限制'])}: {random.choice(['必须满足', '不能超过', '至少达到', '优化到'])} {random.uniform(0.0, 100.0)}"
                for _ in range(random.randint(1, 3))
            ],
            "solution": {
                f"变量{j}": random.uniform(0.0, 100.0)
                for j in range(random.randint(2, 5))
            },
            "source": "synthetic"
        }
        data.append(optimization_data)
    
    return data

def create_synthetic_autonomous_data(size: int) -> List[Dict[str, Any]]:
    """生成合成自主决策数据
    Generate synthetic autonomous decision data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 自主决策数据列表
    """
    data = []
    scenarios = ["导航路径选择", "资源调度", "任务优先级确定", "异常处理", "学习内容选择"]
    
    for i in range(size):
        autonomous_data = {
            "decision_id": f"decision_{i}",
            "scenario": random.choice(scenarios),
            "context": f"在{random.choice(['复杂环境', '时间紧迫', '资源有限', '信息不完整', '动态变化'])}的情况下，需要做出关于{random.choice(['路径规划', '资源分配', '任务执行', '风险评估', '目标调整'])}的决策。",
            "available_options": [
                f"选项{j}: {random.choice(['选择最短路径', '分配更多资源', '优先处理重要任务', '采取保守策略', '尝试创新方案'])}"
                for j in range(random.randint(2, 4))
            ],
            "chosen_option": f"选项{random.randint(0, 3)}: {random.choice(['选择最短路径', '分配更多资源', '优先处理重要任务', '采取保守策略', '尝试创新方案'])}",
            "reasoning": f"基于{random.choice(['当前情况分析', '历史经验', '预期结果评估', '风险考量', '目标优先级'])}, 选择该选项可以{random.choice(['提高效率', '降低风险', '优化资源利用', '加快进度', '提高成功率'])}",
            "outcome": random.choice(["成功", "部分成功", "需要调整", "失败"]),
            "source": "synthetic"
        }
        data.append(autonomous_data)
    
    return data

def create_synthetic_manager_data(size: int) -> List[Dict[str, Any]]:
    """生成合成管理决策数据
    Generate synthetic management decision data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 管理决策数据列表
    """
    data = []
    management_tasks = ["团队管理", "项目规划", "资源分配", "冲突解决", "绩效评估"]
    
    for i in range(size):
        manager_data = {
            "management_id": f"management_{i}",
            "task_type": random.choice(management_tasks),
            "description": f"需要处理关于{random.choice(['团队协作', '项目进度', '资源需求', '人员冲突', '绩效问题'])}的管理事务。",
            "context": f"在{random.choice(['大型项目', '跨部门协作', '时间紧迫', '资源有限', '人员变动'])}的背景下，需要做出管理决策。",
            "decision": f"决定采取{random.choice(['调整团队结构', '重新分配任务', '增加资源投入', '进行沟通协调', '实施激励措施'])}的措施。",
            "expected_outcome": f"希望通过该决策能够{random.choice(['提高团队效率', '加快项目进度', '解决资源问题', '缓解冲突', '提升绩效'])}",
            "actual_outcome": random.choice(["达到预期", "部分达到预期", "未达到预期", "超出预期"]),
            "source": "synthetic"
        }
        data.append(manager_data)
    
    return data

def create_synthetic_value_alignment_data(size: int) -> List[Dict[str, Any]]:
    """生成合成价值观对齐数据
    Generate synthetic value alignment data
    
    Args:
        size: 数据规模
        
    Returns:
        List[Dict[str, Any]]: 价值观对齐数据列表
    """
    data = []
    ethical_principles = ["公平正义", "尊重隐私", "不伤害他人", "诚实可信", "社会责任", "可持续发展"]
    
    for i in range(size):
        value_alignment_data = {
            "alignment_id": f"alignment_{i}",
            "scenario": f"在{random.choice(['自动驾驶', '医疗诊断', '资源分配', '招聘决策', '内容审核'])}场景中，需要考虑伦理问题。",
            "ethical_question": f"是否应该{random.choice(['优先保护多数人', '尊重个人隐私', '考虑长期影响', '遵循程序正义', '平衡各方利益'])}？",
            "principles_involved": random.sample(ethical_principles, k=random.randint(1, 3)),
            "decision": f"决定{random.choice(['优先保护生命安全', '尊重个人选择权', '采取最公平的方案', '遵循法律规定', '考虑社会整体利益'])}",
            "justification": f"该决策符合{', '.join(random.sample(ethical_principles, k=random.randint(1, 2)))}等伦理原则。",
            "alignment_score": random.uniform(0.5, 1.0),
            "source": "synthetic"
        }
        data.append(value_alignment_data)
    
    return data

def generate_training_data_for_all_models() -> Dict[str, Any]:
    """为所有模型生成训练数据
    Generate training data for all models
    
    Returns:
        Dict[str, Any]: 所有模型的训练数据字典，key为模型ID，value为训练数据
    """
    print("开始为所有模型生成训练数据...")
    
    # 根据training_plan.md中的数据集要求生成数据
    training_data = {
        # 基础感知与认知模型
        "knowledge": {
            "type": "text",
            "data": create_synthetic_text_data(10000, data_type="knowledge"),  # 10000条知识文本
            "source": "synthetic",
            "size": 10000
        },
        "language": {
            "type": "text",
            "data": create_synthetic_text_data(5000, data_type="language"),  # 5000条语言文本
            "source": "synthetic",
            "size": 5000
        },
        "vision_image": {
            "type": "image",
            "data": create_synthetic_image_data(1000),  # 1000张图像数据（模拟）
            "source": "synthetic",
            "size": 1000
        },
        "audio": {
            "type": "audio",
            "data": create_synthetic_audio_data(500),  # 500条音频数据（模拟）
            "source": "synthetic",
            "size": 500
        },
        
        # 专业功能模型
        "vision_video": {
            "type": "video",
            "data": create_synthetic_video_data(100),  # 100条视频数据（模拟）
            "source": "synthetic",
            "size": 100
        },
        "spatial": {
            "type": "3d",
            "data": create_synthetic_spatial_data(100),  # 100条空间数据
            "source": "synthetic",
            "size": 100
        },
        "sensor": {
            "type": "sensor",
            "data": create_synthetic_sensor_data(1000),  # 1000条传感器数据
            "source": "synthetic",
            "size": 1000
        },
        "computer": {
            "type": "command_response",
            "data": create_synthetic_computer_data(100),  # 100条命令与响应数据
            "source": "synthetic",
            "size": 100
        },
        "motion": {
            "type": "motion",
            "data": create_synthetic_motion_data(500),  # 500条运动数据
            "source": "synthetic",
            "size": 500
        },
        "programming": {
            "type": "code",
            "data": create_synthetic_programming_data(1000),  # 1000条编程数据
            "source": "synthetic",
            "size": 1000
        },
        "mathematics": {
            "type": "problem_solution",
            "data": create_synthetic_mathematics_data(5000),  # 5000条数学数据
            "source": "synthetic",
            "size": 5000
        },
        
        # 高级认知与决策模型
        "emotion": {
            "type": "text",
            "data": create_synthetic_emotion_data(500),  # 500条情感数据
            "source": "synthetic",
            "size": 500
        },
        "planning": {
            "type": "text",
            "data": create_synthetic_planning_data(100),  # 100条规划数据
            "source": "synthetic",
            "size": 100
        },
        "prediction": {
            "type": "time_series",
            "data": create_synthetic_prediction_data(1000),  # 1000条预测数据
            "source": "synthetic",
            "size": 1000
        },
        "collaboration": {
            "type": "dialogue",
            "data": create_synthetic_collaboration_data(500),  # 500条协作数据
            "source": "synthetic",
            "size": 500
        },
        "optimization": {
            "type": "problem_solution",
            "data": create_synthetic_optimization_data(100),  # 100条优化数据
            "source": "synthetic",
            "size": 100
        },
        
        # 元认知与管理模型
        "autonomous": {
            "type": "decision",
            "data": create_synthetic_autonomous_data(500),  # 500条自主决策数据
            "source": "synthetic",
            "size": 500
        },
        "manager": {
            "type": "management",
            "data": create_synthetic_manager_data(100),  # 100条管理决策数据
            "source": "synthetic",
            "size": 100
        },
        "value_alignment": {
            "type": "ethical",
            "data": create_synthetic_value_alignment_data(500),  # 500条价值观对齐数据
            "source": "synthetic",
            "size": 500
        }
    }
    
    print("所有模型的训练数据生成完成！")
    return training_data

def save_training_data(data: Dict[str, Any], output_dir: str = "training_data") -> None:
    """保存训练数据到文件
    Save training data to files
    
    Args:
        data: 训练数据字典
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    for model_id, model_data in data.items():
        # 创建模型数据目录
        model_dir = os.path.join(output_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存数据
        data_file = os.path.join(model_dir, "training_data.json")
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        print(f"保存模型 {model_id} 的训练数据到 {data_file}")

def main():
    """主函数
    Main function
    """
    # 初始化模型注册表和训练管理器
    print("初始化模型注册表和训练管理器...")
    try:
        # 创建训练准备实例
        training_preparation = create_training_preparation()
        
        if training_preparation is None:
            print("无法创建训练准备实例，程序退出")
            return
        
        # 获取模型注册表和训练管理器
        model_registry = training_preparation.model_registry
        training_manager = training_preparation.training_manager
        
        # 准备环境
        print("\n准备训练环境...")
        env_result = training_preparation.prepare_training_environment()
        if env_result['success']:
            print("环境准备成功！")
        else:
            print(f"环境准备失败: {env_result['message']}")
            return
        
        # 获取所有模型ID
        model_ids = list(model_registry.get_all_models().keys())
        print(f"\n将对以下模型进行训练: {', '.join(model_ids)}")
        
        # 重复生成数据集并训练30次
        for iteration in range(30):
            print(f"\n=== 第 {iteration + 1}/30 次迭代 ===")
            
            # 生成新的训练数据
            print("\n1. 生成新的训练数据...")
            training_data = generate_training_data_for_all_models()
            
            # 保存训练数据
            print("\n2. 保存训练数据...")
            save_training_data(training_data)
            
            # 显示数据统计
            print("\n3. 训练数据统计：")
            for model_id, model_data in training_data.items():
                print(f"- {model_id}: {model_data['size']}条数据 ({model_data['type']})")
            
            # 执行训练
            print("\n4. 开始模型训练...")
            try:
                # 设置训练参数
                training_params = {
                    'training_mode': 'individual',
                    'epochs': 1,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'data_source': 'training_data'
                }
                
                # 启动训练
                job_id = training_manager.start_training(model_ids, training_params)
                print(f"训练作业已启动，作业ID: {job_id}")
                
                # 等待训练完成（由于是线程异步，这里简单等待一段时间）
                print("等待训练完成...")
                time.sleep(60)  # 等待60秒让训练有足够时间执行
                
                # 检查训练状态
                job_status = training_manager.get_job_status(job_id)
                print(f"训练状态: {job_status['status']}")
                if 'message' in job_status:
                    print(f"训练消息: {job_status['message']}")
                    
            except Exception as training_e:
                print(f"训练执行失败: {str(training_e)}")
        
        print("\n=== 30次迭代完成 ===")
        print("所有模型已完成30次新数据生成和训练！")
            
    except Exception as e:
        print(f"初始化失败: {str(e)}")

if __name__ == "__main__":
    main()
