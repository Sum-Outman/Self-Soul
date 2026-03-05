"""
核心调度层演示
展示如何使用新实现的核心调度层解决报告中指出的顶层调度缺失问题
"""

import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.core_scheduling_layer import CoreSchedulingLayer, TaskPriority
from core.model_adapters import ModelAdapterFactory
from core.cross_domain_task_mapping import CrossDomainTaskMapper, DomainType

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DemoModel:
    """演示用模型类"""
    
    def __init__(self, model_id, model_type):
        self.model_id = model_id
        self.model_type = model_type
    
    def process_text(self, text):
        """处理文本"""
        return f"处理结果: {text} (由 {self.model_id} 处理)"
    
    def analyze_image(self, image_data):
        """分析图像"""
        return f"图像分析结果: 检测到对象 (由 {self.model_id} 分析)"
    
    def answer_question(self, question):
        """回答问题"""
        return f"答案: 根据知识库，{question} 的答案是... (由 {self.model_id} 回答)"
    
    def generate_code(self, requirement):
        """生成代码"""
        return f"代码生成: def solve():\n    # {requirement}\n    return '解决方案' (由 {self.model_id} 生成)"
    
    def plan_task(self, task_description):
        """规划任务"""
        return f"任务规划: 分解为3个子任务 (由 {self.model_id} 规划)"
    
    def _get_model_capabilities(self):
        """获取模型能力"""
        return {
            "language_processing": True,
            "vision_analysis": self.model_type == "vision",
            "knowledge_reasoning": self.model_type == "knowledge",
            "programming_code": self.model_type == "programming",
            "planning_scheduling": self.model_type == "manager"
        }


def setup_demo_system():
    """设置演示系统"""
    logger.info("=== 设置核心调度层演示系统 ===")
    
    # 1. 创建核心调度层
    scheduler = CoreSchedulingLayer()
    logger.info("1. 核心调度层创建完成")
    
    # 2. 创建演示模型
    models = [
        ("language_model", "language"),
        ("vision_model", "vision"),
        ("knowledge_model", "knowledge"),
        ("programming_model", "programming"),
        ("manager_model", "manager"),
    ]
    
    # 3. 注册模型到调度层
    for model_id, model_type in models:
        # 创建模型实例
        model_instance = DemoModel(model_id, model_type)
        
        # 创建适配器
        adapter = ModelAdapterFactory.create_adapter(model_instance, f"{model_type}_model")
        
        if adapter:
            scheduler.register_model(model_id, adapter)
            logger.info(f"   注册模型: {model_id} ({model_type})")
        else:
            logger.warning(f"   无法为 {model_id} 创建适配器")
    
    logger.info(f"2. 注册了 {len(models)} 个模型到调度层")
    
    # 4. 创建跨领域映射器
    mapper = CrossDomainTaskMapper()
    logger.info("3. 跨领域任务映射器创建完成")
    
    return scheduler, mapper


def demonstrate_task_processing(scheduler):
    """演示任务处理"""
    logger.info("\n=== 演示任务处理 ===")
    
    # 测试案例
    test_cases = [
        {
            "id": "task_1",
            "description": "分析这张图片中的物体并生成描述文本",
            "priority": TaskPriority.HIGH,
            "expected_domains": ["vision", "language"]
        },
        {
            "id": "task_2",
            "description": "设计一个医疗监测设备的机械结构并编写控制代码",
            "priority": TaskPriority.CRITICAL,
            "expected_domains": ["engineering", "medical", "programming"]
        },
        {
            "id": "task_3",
            "description": "规划一个跨领域研究项目，包含工程和医疗组件",
            "priority": TaskPriority.HIGH,
            "expected_domains": ["manager", "engineering", "medical"]
        },
        {
            "id": "task_4",
            "description": "简单文本处理任务",
            "priority": TaskPriority.LOW,
            "expected_domains": ["language"]
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\n处理任务: {test_case['id']}")
        logger.info(f"描述: {test_case['description']}")
        logger.info(f"优先级: {test_case['priority'].value}")
        logger.info(f"预期领域: {test_case['expected_domains']}")
        
        # 处理任务
        result = scheduler.process_task(
            task_id=test_case['id'],
            description=test_case['description'],
            priority=test_case['priority']
        )
        
        # 显示结果摘要
        summary = result['summary']
        logger.info(f"结果摘要:")
        logger.info(f"  成功: {summary['success']}")
        logger.info(f"  质量评分: {summary['quality_score']:.2f}")
        logger.info(f"  总执行时间: {summary['total_execution_time']:.2f}秒")
        logger.info(f"  完成步骤: {summary['steps_completed']}")
        
        # 显示验证结果
        validation = result['validation_result']
        logger.info(f"验证结果:")
        logger.info(f"  完整性: {validation.completeness_score:.2f}")
        logger.info(f"  正确性: {validation.correctness_score:.2f}")
        logger.info(f"  效率: {validation.efficiency_score:.2f}")
        
        if validation.improvement_suggestions:
            logger.info(f"改进建议: {validation.improvement_suggestions[0]}")


def demonstrate_cross_domain_mapping(mapper):
    """演示跨领域映射"""
    logger.info("\n=== 演示跨领域映射 ===")
    
    # 测试跨领域任务
    cross_domain_tasks = [
        "设计一个智能医疗机器人系统",
        "分析金融数据并生成工程投资建议",
        "开发教育用编程学习工具",
        "创建创意艺术与工程技术结合的作品"
    ]
    
    for task_description in cross_domain_tasks:
        logger.info(f"\n分析任务: {task_description}")
        
        # 识别领域
        domains = mapper.identify_domains(task_description)
        logger.info(f"  识别到的领域: {[d.value for d in domains]}")
        
        # 生成跨领域计划
        base_capabilities = [DomainType.ENGINEERING] if domains else []
        plan = mapper.generate_cross_domain_plan(
            task_description, domains, base_capabilities
        )
        
        logger.info(f"  跨领域计划:")
        logger.info(f"    是否跨领域: {plan['cross_domain']}")
        logger.info(f"    基础能力: {plan.get('base_capabilities', [])}")
        logger.info(f"    增强能力: {plan.get('enhanced_capabilities', [])}")
        logger.info(f"    复杂度因子: {plan.get('complexity_factor', 1.0):.2f}")
        
        # 分析复杂度
        if domains:
            complexity = mapper.analyze_task_complexity(task_description, domains)
            logger.info(f"  任务复杂度: {complexity:.2f}")
            
            if complexity > 0.7:
                logger.info("  分类: 高复杂度任务")
            elif complexity > 0.4:
                logger.info("  分类: 中等复杂度任务")
            else:
                logger.info("  分类: 低复杂度任务")


def demonstrate_system_integration(scheduler, mapper):
    """演示系统集成"""
    logger.info("\n=== 演示系统集成 ===")
    
    # 获取系统状态
    status = scheduler.get_system_status()
    logger.info("系统状态:")
    logger.info(f"  已注册模型: {status['registered_models']}")
    logger.info(f"  可用能力: {status['available_capabilities']}")
    logger.info(f"  待处理任务: {status['pending_tasks']}")
    logger.info(f"  已完成任务: {status['completed_tasks']}")
    logger.info(f"  平均质量评分: {status['average_quality_score']:.2f}")
    
    # 获取映射统计
    mapper_stats = mapper.get_mapping_statistics()
    logger.info("跨领域映射统计:")
    logger.info(f"  领域配置数: {mapper_stats['total_domain_profiles']}")
    logger.info(f"  跨领域映射数: {mapper_stats['total_cross_domain_mappings']}")
    logger.info(f"  映射使用次数: {mapper_stats['total_mapping_usage']}")
    logger.info(f"  平均成功率: {mapper_stats['average_success_rate']:.2f}")
    
    # 演示复杂任务处理
    logger.info("\n演示复杂跨领域任务处理:")
    complex_task = {
        "id": "complex_demo_task",
        "description": "开发一个智能健康监测系统，需要机械设计、医疗数据分析、软件编程和系统集成",
        "priority": TaskPriority.CRITICAL
    }
    
    logger.info(f"任务: {complex_task['description']}")
    
    # 使用跨领域映射器分析
    domains = mapper.identify_domains(complex_task['description'])
    logger.info(f"跨领域分析识别: {[d.value for d in domains]}")
    
    # 处理任务
    result = scheduler.process_task(
        task_id=complex_task['id'],
        description=complex_task['description'],
        priority=complex_task['priority']
    )
    
    logger.info(f"处理结果:")
    logger.info(f"  成功: {result['summary']['success']}")
    logger.info(f"  质量: {result['summary']['quality_score']:.2f}")
    logger.info(f"  步骤数: {result['summary']['steps_completed']}")
    
    # 显示调度决策
    if 'execution_result' in result and 'results' in result['execution_result']:
        results = result['execution_result']['results']
        logger.info(f"  调度决策:")
        for step_key, step_result in results.items():
            if isinstance(step_result, dict) and step_result.get('success'):
                logger.info(f"    {step_key}: {step_result.get('capability')} -> {step_result.get('model')}")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("核心调度层演示")
    logger.info("解决报告中指出的顶层调度缺失问题")
    logger.info("=" * 60)
    
    try:
        # 设置系统
        scheduler, mapper = setup_demo_system()
        
        # 演示任务处理
        demonstrate_task_processing(scheduler)
        
        # 演示跨领域映射
        demonstrate_cross_domain_mapping(mapper)
        
        # 演示系统集成
        demonstrate_system_integration(scheduler, mapper)
        
        logger.info("\n" + "=" * 60)
        logger.info("演示完成")
        logger.info("核心调度层成功解决了以下问题:")
        logger.info("1. 顶层调度缺失 - 通过任务解析器+能力调度中枢+结果验证器")
        logger.info("2. 跨领域协同 - 通过跨领域任务映射规则")
        logger.info("3. 动态能力组合 - 通过统一能力接口标准")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())