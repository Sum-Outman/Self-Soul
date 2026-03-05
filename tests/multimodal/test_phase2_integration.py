"""
多模态第二阶段集成测试

验证修复计划第二阶段的所有组件：
1. HybridModalParser - 混合模态解析
2. IntentFusionEngine - 意图融合
3. FaultToleranceManager - 容错处理

目标：验证混合输入处理成功率>90%
"""

import sys
import os
import json
import time
from typing import Dict, Any, List
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入多模态组件
from core.multimodal.hybrid_modal_parser import HybridModalParser
from core.multimodal.intent_fusion_engine import IntentFusionEngine
from core.multimodal.fault_tolerance_manager import FaultToleranceManager, ModalityType


class Phase2IntegrationTest:
    """第二阶段集成测试"""
    
    def __init__(self):
        """初始化测试"""
        self.parser = HybridModalParser()
        self.intent_engine = IntentFusionEngine()
        self.fault_tolerance_manager = FaultToleranceManager()
        
        # 测试统计
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
        
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("=" * 80)
        print("多模态第二阶段集成测试")
        print("目标：验证混合输入处理成功率 > 90%")
        print("=" * 80)
        
        # 运行测试套件
        test_methods = [
            self.test_hybrid_parser_basic,
            self.test_hybrid_parser_mixed_modalities,
            self.test_hybrid_parser_low_quality,
            self.test_intent_fusion_basic,
            self.test_intent_fusion_complementary_intent,
            self.test_intent_fusion_context_memory,
            self.test_fault_tolerance_text,
            self.test_fault_tolerance_image,
            self.test_fault_tolerance_audio,
            self.test_end_to_end_scenario_1,
            self.test_end_to_end_scenario_2,
            self.test_end_to_end_scenario_3,
            self.test_performance_benchmark
        ]
        
        for test_method in test_methods:
            self._run_test(test_method)
        
        # 计算成功率
        success_rate = (self.test_results["passed_tests"] / 
                       max(self.test_results["total_tests"], 1)) * 100
        
        # 打印总结
        print("\n" + "=" * 80)
        print("测试总结")
        print("=" * 80)
        print(f"总测试数: {self.test_results['total_tests']}")
        print(f"通过数: {self.test_results['passed_tests']}")
        print(f"失败数: {self.test_results['failed_tests']}")
        print(f"成功率: {success_rate:.2f}%")
        print(f"目标: > 90%")
        print(f"结果: {'✓ 通过' if success_rate > 90 else '✗ 失败'}")
        
        # 详细结果
        if self.test_results["failed_tests"] > 0:
            print("\n失败测试详情:")
            for detail in self.test_results["test_details"]:
                if not detail["passed"]:
                    print(f"  - {detail['name']}: {detail.get('error', '未知错误')}")
        
        return {
            "success_rate": success_rate,
            "passed": success_rate > 90,
            "details": self.test_results
        }
    
    def _run_test(self, test_method) -> None:
        """运行单个测试"""
        test_name = test_method.__name__
        print(f"\n[测试] {test_name}")
        
        try:
            start_time = time.time()
            result = test_method()
            elapsed_time = time.time() - start_time
            
            if result.get("passed", False):
                self.test_results["passed_tests"] += 1
                print(f"  ✓ 通过 ({elapsed_time:.2f}s)")
            else:
                self.test_results["failed_tests"] += 1
                print(f"  ✗ 失败 ({elapsed_time:.2f}s): {result.get('error', '未知错误')}")
            
            self.test_results["total_tests"] += 1
            self.test_results["test_details"].append({
                "name": test_name,
                "passed": result.get("passed", False),
                "elapsed_time": elapsed_time,
                "error": result.get("error"),
                "details": result.get("details")
            })
            
        except Exception as e:
            self.test_results["failed_tests"] += 1
            self.test_results["total_tests"] += 1
            print(f"  ✗ 异常: {e}")
            self.test_results["test_details"].append({
                "name": test_name,
                "passed": False,
                "error": str(e)
            })
    
    # ===== 混合模态解析器测试 =====
    
    def test_hybrid_parser_basic(self) -> Dict[str, Any]:
        """测试混合模态解析器基本功能"""
        test_input = {
            "text": "这是一个测试文本",
            "image_data": "模拟图像数据",
            "metadata": {"source": "test"}
        }
        
        result = self.parser.parse_hybrid_input(test_input)
        
        return {
            "passed": result.get("success", False),
            "details": {
                "parsed_modalities": result.get("parsed_modalities", []),
                "original_keys": result.get("original_input_keys", [])
            }
        }
    
    def test_hybrid_parser_mixed_modalities(self) -> Dict[str, Any]:
        """测试混合模态解析（多种模态）"""
        test_input = {
            "text": "模糊图片描述",
            "image_data": "模糊的JPEG图像数据",
            "audio_data": "嘈杂的音频数据",
            "metadata": {
                "source": "test",
                "image_quality": 0.3,
                "audio_quality": 0.4
            }
        }
        
        result = self.parser.parse_hybrid_input(test_input)
        
        # 检查是否成功解析了所有模态
        parsed_modalities = result.get("parsed_modalities", [])
        has_text = "text" in parsed_modalities
        has_image = "image" in parsed_modalities
        has_audio = "audio" in parsed_modalities
        
        return {
            "passed": result.get("success", False) and len(parsed_modalities) >= 2,
            "details": {
                "parsed_modalities": parsed_modalities,
                "quality_assessments": result.get("quality_assessments", {})
            }
        }
    
    def test_hybrid_parser_low_quality(self) -> Dict[str, Any]:
        """测试低质量输入解析"""
        test_input = {
            "text": "thsi is a txt with speling erors",
            "image_data": "非常模糊的图像",
            "metadata": {
                "source": "test",
                "image_quality": 0.1,
                "format_warning": True
            }
        }
        
        result = self.parser.parse_hybrid_input(test_input)
        
        # 检查修复报告
        repair_reports = result.get("repair_reports", {})
        text_repaired = "text" in repair_reports
        image_repaired = "image" in repair_reports
        
        return {
            "passed": result.get("success", False) and (text_repaired or image_repaired),
            "details": {
                "repair_reports": repair_reports,
                "noise_detections": result.get("noise_detections", {})
            }
        }
    
    # ===== 意图融合引擎测试 =====
    
    def test_intent_fusion_basic(self) -> Dict[str, Any]:
        """测试意图融合引擎基本功能"""
        test_input = {
            "text": "今天天气怎么样？",
            "metadata": {"source": "test"}
        }
        
        result = self.intent_engine.process_multimodal_input(test_input)
        
        return {
            "passed": result.get("success", False),
            "details": {
                "intent_elements": len(result.get("intent_elements", [])),
                "fused_intent": result.get("fused_intent", {})
            }
        }
    
    def test_intent_fusion_complementary_intent(self) -> Dict[str, Any]:
        """测试互补意图融合"""
        test_input = {
            "text": "修这个需要哪些零件",
            "image_data": "破损键盘图片",
            "metadata": {"source": "test"}
        }
        
        result = self.intent_engine.process_multimodal_input(test_input)
        fused_intent = result.get("fused_intent", {})
        
        # 检查是否成功融合
        intent_type = fused_intent.get("intent_type", "")
        fused_description = fused_intent.get("fused_description", "")
        has_fusion = "修复" in fused_description or "图片" in fused_description
        
        return {
            "passed": result.get("success", False) and has_fusion,
            "details": {
                "intent_type": intent_type,
                "fused_description": fused_description,
                "fusion_quality": fused_intent.get("fusion_quality", 0)
            }
        }
    
    def test_intent_fusion_context_memory(self) -> Dict[str, Any]:
        """测试上下文记忆"""
        # 第一轮对话
        input_1 = {
            "text": "这张图片里的猫咪是什么品种？",
            "image_data": "猫咪图片",
            "metadata": {"source": "test", "conversation_id": "test_conv_1"}
        }
        
        result_1 = self.intent_engine.process_multimodal_input(input_1)
        context_id = result_1.get("fused_intent", {}).get("context_id")
        
        # 第二轮对话（使用上下文）
        input_2 = {
            "text": "它多大了？",
            "metadata": {"source": "test", "conversation_id": "test_conv_1"}
        }
        
        result_2 = self.intent_engine.process_multimodal_input(input_2, context_id)
        fused_intent = result_2.get("fused_intent", {})
        
        # 检查是否使用了上下文
        context_used = fused_intent.get("context_used") is not None
        
        return {
            "passed": result_2.get("success", False) and context_used,
            "details": {
                "context_id": context_id,
                "context_used": fused_intent.get("context_used"),
                "explanation": fused_intent.get("explanation", "")
            }
        }
    
    # ===== 容错管理器测试 =====
    
    def test_fault_tolerance_text(self) -> Dict[str, Any]:
        """测试文本容错处理"""
        test_input = {
            "text": "thsi is brokn txt with many erors",
            "metadata": {"source": "test"}
        }
        
        result = self.fault_tolerance_manager.process_multimodal_input(test_input)
        text_result = result.get("modality_results", {}).get("text", {})
        
        # 检查是否进行了修复
        status = text_result.get("status", "")
        has_repair = status in ["repaired", "optimized"]
        quality_improvement = text_result.get("quality_improvement", 0)
        
        return {
            "passed": text_result.get("success", False) and has_repair and quality_improvement > 0,
            "details": {
                "status": status,
                "quality_improvement": quality_improvement,
                "original_quality": text_result.get("original_quality", {}),
                "final_quality": text_result.get("final_quality", {})
            }
        }
    
    def test_fault_tolerance_image(self) -> Dict[str, Any]:
        """测试图像容错处理"""
        test_input = {
            "image_data": "低质量图像数据",
            "metadata": {
                "source": "test",
                "image_quality": 0.2,
                "corruption_detected": True
            }
        }
        
        result = self.fault_tolerance_manager.process_multimodal_input(test_input)
        image_result = result.get("modality_results", {}).get("image", {})
        
        # 检查是否应用了降级策略
        degradation_applied = "degradation_applied" in image_result
        status = image_result.get("status", "")
        
        return {
            "passed": image_result.get("success", False) and 
                     (status == "degraded" or degradation_applied),
            "details": {
                "status": status,
                "degradation_applied": image_result.get("degradation_applied"),
                "quality_improvement": image_result.get("quality_improvement", 0)
            }
        }
    
    def test_fault_tolerance_audio(self) -> Dict[str, Any]:
        """测试音频容错处理"""
        test_input = {
            "audio_data": "嘈杂音频数据",
            "metadata": {
                "source": "test",
                "audio_quality": 0.3,
                "noise_level": "high"
            }
        }
        
        result = self.fault_tolerance_manager.process_multimodal_input(test_input)
        audio_result = result.get("modality_results", {}).get("audio", {})
        
        # 检查是否进行了处理
        status = audio_result.get("status", "")
        processed = status in ["repaired", "optimized", "degraded"]
        
        return {
            "passed": audio_result.get("success", False) and processed,
            "details": {
                "status": status,
                "repair_history": audio_result.get("repair_history", []),
                "quality_improvement": audio_result.get("quality_improvement", 0)
            }
        }
    
    # ===== 端到端场景测试 =====
    
    def test_end_to_end_scenario_1(self) -> Dict[str, Any]:
        """端到端场景1：修复请求"""
        # 用户上传破损键盘图片并询问修复方法
        test_input = {
            "text": "hw to fix this brokn keyboard? it has missing keys.",
            "image_data": "破损键盘图片，键帽缺失",
            "metadata": {
                "source": "user_upload",
                "image_quality": 0.4,
                "has_damage": True
            }
        }
        
        # 1. 解析混合输入
        parse_result = self.parser.parse_hybrid_input(test_input)
        
        # 2. 融合意图
        intent_result = self.intent_engine.process_multimodal_input(test_input)
        
        # 3. 容错处理
        tolerance_result = self.fault_tolerance_manager.process_multimodal_input(test_input)
        
        # 检查所有步骤都成功
        parse_success = parse_result.get("success", False)
        intent_success = intent_result.get("success", False)
        tolerance_success = tolerance_result.get("success", False)
        
        # 检查意图融合是否识别了修复请求
        fused_intent = intent_result.get("fused_intent", {})
        intent_type = fused_intent.get("intent_type", "")
        has_repair_intent = "repair" in intent_type.lower() or "修复" in fused_intent.get("fused_description", "")
        
        return {
            "passed": parse_success and intent_success and tolerance_success and has_repair_intent,
            "details": {
                "parse_success": parse_success,
                "intent_success": intent_success,
                "tolerance_success": tolerance_success,
                "intent_type": intent_type,
                "fused_description": fused_intent.get("fused_description", "")
            }
        }
    
    def test_end_to_end_scenario_2(self) -> Dict[str, Any]:
        """端到端场景2：多模态查询"""
        # 用户在嘈杂环境中询问天气，同时上传阴天图片
        test_input = {
            "text": "what's the weather today?",
            "image_data": "阴天图片，云层厚",
            "audio_data": "背景风声嘈杂",
            "metadata": {
                "source": "mobile_app",
                "image_quality": 0.7,
                "audio_quality": 0.5,
                "environment": "noisy"
            }
        }
        
        # 1. 解析混合输入
        parse_result = self.parser.parse_hybrid_input(test_input)
        
        # 2. 融合意图
        intent_result = self.intent_engine.process_multimodal_input(test_input)
        
        # 3. 容错处理
        tolerance_result = self.fault_tolerance_manager.process_multimodal_input(test_input)
        
        # 检查所有步骤都成功
        parse_success = parse_result.get("success", False)
        intent_success = intent_result.get("success", False)
        tolerance_success = tolerance_result.get("success", False)
        
        # 检查是否处理了所有模态
        parsed_modalities = parse_result.get("parsed_modalities", [])
        has_all_modalities = len(parsed_modalities) >= 2
        
        return {
            "passed": parse_success and intent_success and tolerance_success and has_all_modalities,
            "details": {
                "parse_success": parse_success,
                "intent_success": intent_success,
                "tolerance_success": tolerance_success,
                "parsed_modalities": parsed_modalities,
                "modality_results": list(tolerance_result.get("modality_results", {}).keys())
            }
        }
    
    def test_end_to_end_scenario_3(self) -> Dict[str, Any]:
        """端到端场景3：低质量混合输入"""
        # 用户上传模糊图片、嘈杂语音和有错别字的文本
        test_input = {
            "text": "pls describ this piture, its very blury",
            "image_data": "极度模糊的图片，无法识别细节",
            "audio_data": "背景噪音极大的语音",
            "metadata": {
                "source": "low_quality_upload",
                "image_quality": 0.1,
                "audio_quality": 0.2,
                "format_unsupported": False,
                "corruption_detected": True
            }
        }
        
        # 1. 解析混合输入
        parse_result = self.parser.parse_hybrid_input(test_input)
        
        # 2. 容错处理
        tolerance_result = self.fault_tolerance_manager.process_multimodal_input(test_input)
        
        # 检查解析是否成功（即使质量低）
        parse_success = parse_result.get("success", False)
        
        # 检查容错处理是否应用了修复或降级
        modality_results = tolerance_result.get("modality_results", {})
        has_repairs_or_degradations = False
        
        for modality, result in modality_results.items():
            status = result.get("status", "")
            if status in ["repaired", "optimized", "degraded"]:
                has_repairs_or_degradations = True
                break
        
        return {
            "passed": parse_success and tolerance_result.get("success", False) and has_repairs_or_degradations,
            "details": {
                "parse_success": parse_success,
                "tolerance_success": tolerance_result.get("success", False),
                "modality_statuses": {mod: res.get("status", "") for mod, res in modality_results.items()},
                "repair_counts": sum(1 for res in modality_results.values() if res.get("status") == "repaired")
            }
        }
    
    # ===== 性能基准测试 =====
    
    def test_performance_benchmark(self) -> Dict[str, Any]:
        """性能基准测试"""
        test_cases = [
            {
                "name": "纯文本",
                "input": {
                    "text": "这是一个简单的测试文本，用于性能基准测试。",
                    "metadata": {"source": "benchmark"}
                }
            },
            {
                "name": "文本+图像",
                "input": {
                    "text": "描述这张图片的内容。",
                    "image_data": "标准测试图像",
                    "metadata": {"source": "benchmark", "image_quality": 0.8}
                }
            },
            {
                "name": "三模态混合",
                "input": {
                    "text": "综合分析这个场景。",
                    "image_data": "测试图像",
                    "audio_data": "测试音频",
                    "metadata": {"source": "benchmark", "image_quality": 0.7, "audio_quality": 0.6}
                }
            },
            {
                "name": "低质量混合",
                "input": {
                    "text": "thsi is low qality txt",
                    "image_data": "模糊图像",
                    "audio_data": "嘈杂音频",
                    "metadata": {
                        "source": "benchmark",
                        "image_quality": 0.3,
                        "audio_quality": 0.4,
                        "corruption_detected": True
                    }
                }
            }
        ]
        
        performance_results = []
        all_successful = True
        
        for test_case in test_cases:
            name = test_case["name"]
            test_input = test_case["input"]
            
            # 测量端到端处理时间
            start_time = time.time()
            
            # 运行完整流程
            parse_result = self.parser.parse_hybrid_input(test_input)
            intent_result = self.intent_engine.process_multimodal_input(test_input)
            tolerance_result = self.fault_tolerance_manager.process_multimodal_input(test_input)
            
            elapsed_time = time.time() - start_time
            
            # 检查成功状态
            parse_success = parse_result.get("success", False)
            intent_success = intent_result.get("success", False)
            tolerance_success = tolerance_result.get("success", False)
            successful = parse_success and intent_success and tolerance_success
            
            if not successful:
                all_successful = False
            
            performance_results.append({
                "name": name,
                "elapsed_time": elapsed_time,
                "successful": successful,
                "modality_count": len([k for k in test_input.keys() if k not in ["metadata"]]),
                "parse_success": parse_success,
                "intent_success": intent_success,
                "tolerance_success": tolerance_success
            })
            
            print(f"  {name}: {elapsed_time:.3f}s, 成功: {successful}")
        
        # 计算平均处理时间
        successful_tests = [r for r in performance_results if r["successful"]]
        if successful_tests:
            avg_time = sum(r["elapsed_time"] for r in successful_tests) / len(successful_tests)
            max_time = max(r["elapsed_time"] for r in successful_tests)
        else:
            avg_time = 0
            max_time = 0
        
        # 检查性能标准：平均处理时间 < 2秒，最大处理时间 < 5秒
        performance_passed = avg_time < 2.0 and max_time < 5.0
        
        return {
            "passed": all_successful and performance_passed,
            "details": {
                "performance_results": performance_results,
                "avg_time": avg_time,
                "max_time": max_time,
                "performance_standard_met": performance_passed,
                "all_successful": all_successful
            }
        }


def main():
    """主函数"""
    print("多模态第二阶段集成测试启动...")
    
    # 创建测试实例
    tester = Phase2IntegrationTest()
    
    # 运行所有测试
    result = tester.run_all_tests()
    
    # 输出详细报告
    if result.get("passed", False):
        print("\n🎉 第二阶段集成测试通过！")
        print(f"混合输入处理成功率: {result['success_rate']:.2f}% (> 90% 目标)")
    else:
        print("\n❌ 第二阶段集成测试失败！")
        print(f"混合输入处理成功率: {result['success_rate']:.2f}% (需要 > 90%)")
    
    # 保存测试结果
    output_file = "phase2_integration_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试结果已保存到: {output_file}")
    
    # 返回退出码
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    sys.exit(main())