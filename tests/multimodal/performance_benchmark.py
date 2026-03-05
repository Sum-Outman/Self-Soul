"""
多模态性能基准测试

验证修复计划第二阶段的性能要求：
1. 确保处理延迟在可接受范围内
2. 验证多模态处理耗时 < 单模态1.5倍（最终目标）
3. 提供详细的性能分析报告
"""

import sys
import os
import time
import json
import statistics
from typing import Dict, Any, List, Tuple
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入多模态组件
from core.multimodal.hybrid_modal_parser import HybridModalParser
from core.multimodal.intent_fusion_engine import IntentFusionEngine
from core.multimodal.fault_tolerance_manager import FaultToleranceManager, ModalityType


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, num_iterations: int = 10):
        """初始化性能基准测试"""
        self.num_iterations = num_iterations
        
        # 初始化组件
        self.parser = HybridModalParser()
        self.intent_engine = IntentFusionEngine()
        self.fault_tolerance_manager = FaultToleranceManager()
        
        # 性能结果
        self.results = {
            "test_config": {
                "num_iterations": num_iterations,
                "timestamp": time.time()
            },
            "single_modality_tests": {},
            "multimodal_tests": {},
            "comparative_analysis": {},
            "summary": {}
        }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """运行完整的性能基准测试"""
        print("=" * 80)
        print("多模态性能基准测试")
        print("目标：验证处理延迟在可接受范围内")
        print("=" * 80)
        
        # 1. 单模态基准测试
        print("\n1. 单模态基准测试...")
        self._run_single_modality_benchmarks()
        
        # 2. 多模态基准测试
        print("\n2. 多模态基准测试...")
        self._run_multimodal_benchmarks()
        
        # 3. 比较分析
        print("\n3. 比较分析...")
        self._perform_comparative_analysis()
        
        # 4. 生成总结报告
        print("\n4. 生成性能总结...")
        self._generate_summary()
        
        return self.results
    
    def _run_single_modality_benchmarks(self) -> None:
        """运行单模态基准测试"""
        # 定义测试用例
        test_cases = [
            ("纯文本高质量", {
                "text": "这是一个高质量的测试文本，包含完整的句子结构和正确的语法。用于评估文本处理性能。",
                "metadata": {"source": "benchmark", "quality": "high"}
            }),
            ("纯文本低质量", {
                "text": "thsi is low qality txt with speling erors and bad grammer",
                "metadata": {"source": "benchmark", "quality": "low"}
            }),
            ("图像高质量", {
                "image_data": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 100,  # 模拟高质量JPEG图像
                "metadata": {"source": "benchmark", "image_quality": 0.9}
            }),
            ("图像低质量", {
                "image_data": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 50,  # 模拟低质量JPEG图像（数据较少）
                "metadata": {"source": "benchmark", "image_quality": 0.2}
            }),
            ("音频高质量", {
                "audio_data": b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 100,  # 模拟高质量WAV音频
                "metadata": {"source": "benchmark", "audio_quality": 0.85}
            }),
            ("音频低质量", {
                "audio_data": b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 50,  # 模拟低质量WAV音频（数据较少）
                "metadata": {"source": "benchmark", "audio_quality": 0.3}
            })
        ]
        
        for name, test_input in test_cases:
            print(f"  测试: {name}")
            
            # 测量解析器性能
            parse_times = self._measure_performance(
                lambda: self.parser.parse_hybrid_input(test_input),
                self.num_iterations
            )
            
            # 测量意图融合性能（如果适用）
            if "text" in test_input:
                intent_times = self._measure_performance(
                    lambda: self.intent_engine.process_multimodal_input(test_input),
                    self.num_iterations
                )
            else:
                intent_times = []
            
            # 测量容错处理性能
            tolerance_times = self._measure_performance(
                lambda: self.fault_tolerance_manager.process_multimodal_input(test_input),
                self.num_iterations
            )
            
            # 保存结果
            self.results["single_modality_tests"][name] = {
                "parse_times": parse_times,
                "intent_times": intent_times,
                "tolerance_times": tolerance_times,
                "avg_parse_time": statistics.mean(parse_times) if parse_times else 0,
                "avg_intent_time": statistics.mean(intent_times) if intent_times else 0,
                "avg_tolerance_time": statistics.mean(tolerance_times) if tolerance_times else 0,
                "total_avg_time": statistics.mean(parse_times + tolerance_times)
            }
    
    def _run_multimodal_benchmarks(self) -> None:
        """运行多模态基准测试"""
        # 定义测试用例
        test_cases = [
            ("双模态高质量", {
                "text": "描述这张高质量图片中的场景和物体。",
                "image_data": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 200,  # 模拟双模态高质量图像
                "metadata": {
                    "source": "benchmark",
                    "image_quality": 0.8,
                    "text_quality": "high"
                }
            }),
            ("三模态高质量", {
                "text": "综合分析这个多模态场景的内容和含义。",
                "image_data": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 250,  # 模拟三模态高质量图像
                "audio_data": b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 200,  # 模拟三模态高质量音频
                "metadata": {
                    "source": "benchmark",
                    "image_quality": 0.85,
                    "audio_quality": 0.8,
                    "text_quality": "high"
                }
            }),
            ("双模态低质量", {
                "text": "thsi is blury piture need describ",
                "image_data": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 80,  # 模拟模糊低质量图像
                "metadata": {
                    "source": "benchmark",
                    "image_quality": 0.3,
                    "text_quality": "low",
                    "corruption_detected": True
                }
            }),
            ("三模态混合质量", {
                "text": "综合分析这个混合质量场景。",
                "image_data": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 120,  # 模拟混合质量图像
                "audio_data": b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 150,  # 模拟混合质量音频
                "metadata": {
                    "source": "benchmark",
                    "image_quality": 0.4,
                    "audio_quality": 0.6,
                    "text_quality": "medium"
                }
            }),
            ("复杂场景修复", {
                "text": "hw to repare this brokn device? it has missing parts and damage.",
                "image_data": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 60,  # 模拟损坏设备图像
                "audio_data": b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 80,  # 模拟背景噪声音频
                "metadata": {
                    "source": "benchmark",
                    "image_quality": 0.2,
                    "audio_quality": 0.3,
                    "text_quality": "low",
                    "has_damage": True,
                    "repair_needed": True
                }
            })
        ]
        
        for name, test_input in test_cases:
            print(f"  测试: {name}")
            
            # 测量端到端性能
            def run_full_pipeline():
                # 解析混合输入
                parse_result = self.parser.parse_hybrid_input(test_input)
                # 融合意图
                intent_result = self.intent_engine.process_multimodal_input(test_input)
                # 容错处理
                tolerance_result = self.fault_tolerance_manager.process_multimodal_input(test_input)
                return parse_result, intent_result, tolerance_result
            
            pipeline_times = self._measure_performance(run_full_pipeline, self.num_iterations)
            
            # 分别测量组件性能
            parse_times = self._measure_performance(
                lambda: self.parser.parse_hybrid_input(test_input),
                self.num_iterations
            )
            
            intent_times = self._measure_performance(
                lambda: self.intent_engine.process_multimodal_input(test_input),
                self.num_iterations
            )
            
            tolerance_times = self._measure_performance(
                lambda: self.fault_tolerance_manager.process_multimodal_input(test_input),
                self.num_iterations
            )
            
            # 保存结果
            self.results["multimodal_tests"][name] = {
                "pipeline_times": pipeline_times,
                "parse_times": parse_times,
                "intent_times": intent_times,
                "tolerance_times": tolerance_times,
                "avg_pipeline_time": statistics.mean(pipeline_times) if pipeline_times else 0,
                "avg_parse_time": statistics.mean(parse_times) if parse_times else 0,
                "avg_intent_time": statistics.mean(intent_times) if intent_times else 0,
                "avg_tolerance_time": statistics.mean(tolerance_times) if tolerance_times else 0,
                "modality_count": len([k for k in test_input.keys() if k not in ["metadata"]]),
                "quality_profile": {
                    "text_quality": test_input.get("metadata", {}).get("text_quality", "unknown"),
                    "image_quality": test_input.get("metadata", {}).get("image_quality", 0),
                    "audio_quality": test_input.get("metadata", {}).get("audio_quality", 0)
                }
            }
    
    def _measure_performance(self, func, iterations: int) -> List[float]:
        """测量函数性能"""
        times = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            result = func()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            # 预热后跳过第一次测量
            if i == 0 and iterations > 1:
                times = []
                start_time = time.perf_counter()
                result = func()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        return times
    
    def _perform_comparative_analysis(self) -> None:
        """执行比较分析"""
        # 计算单模态基准
        text_high_times = self.results["single_modality_tests"]["纯文本高质量"]["total_avg_time"]
        text_low_times = self.results["single_modality_tests"]["纯文本低质量"]["total_avg_time"]
        
        # 计算多模态与单模态的比率
        comparative_data = {}
        
        for test_name, test_data in self.results["multimodal_tests"].items():
            modality_count = test_data["modality_count"]
            avg_pipeline_time = test_data["avg_pipeline_time"]
            
            # 计算与单模态的比率
            if modality_count == 2:
                # 双模态：与文本高质量比较
                ratio_vs_single = avg_pipeline_time / text_high_times if text_high_times > 0 else 0
                comparative_data[test_name] = {
                    "modality_count": modality_count,
                    "avg_time": avg_pipeline_time,
                    "ratio_vs_single_text": ratio_vs_single,
                    "performance_class": self._classify_performance(ratio_vs_single)
                }
            elif modality_count == 3:
                # 三模态：与文本高质量比较
                ratio_vs_single = avg_pipeline_time / text_high_times if text_high_times > 0 else 0
                comparative_data[test_name] = {
                    "modality_count": modality_count,
                    "avg_time": avg_pipeline_time,
                    "ratio_vs_single_text": ratio_vs_single,
                    "performance_class": self._classify_performance(ratio_vs_single)
                }
        
        # 计算质量影响
        quality_impact = {}
        for test_name, test_data in self.results["multimodal_tests"].items():
            quality_profile = test_data["quality_profile"]
            avg_time = test_data["avg_pipeline_time"]
            
            # 根据质量分类
            quality_score = (
                (1.0 if quality_profile.get("text_quality") == "high" else 0.5) +
                quality_profile.get("image_quality", 0) +
                quality_profile.get("audio_quality", 0)
            ) / 3
            
            quality_impact[test_name] = {
                "quality_score": quality_score,
                "processing_time": avg_time,
                "efficiency": quality_score / max(avg_time, 0.001)
            }
        
        self.results["comparative_analysis"] = {
            "single_modality_baseline": {
                "text_high_quality": text_high_times,
                "text_low_quality": text_low_times
            },
            "multimodal_ratios": comparative_data,
            "quality_impact": quality_impact,
            "performance_thresholds": {
                "excellent": {"max_ratio": 1.2, "max_time": 1.0},
                "good": {"max_ratio": 1.5, "max_time": 2.0},
                "acceptable": {"max_ratio": 2.0, "max_time": 3.0},
                "poor": {"max_ratio": 3.0, "max_time": 5.0}
            }
        }
    
    def _classify_performance(self, ratio: float) -> str:
        """分类性能"""
        if ratio <= 1.2:
            return "excellent"
        elif ratio <= 1.5:
            return "good"
        elif ratio <= 2.0:
            return "acceptable"
        elif ratio <= 3.0:
            return "poor"
        else:
            return "unacceptable"
    
    def _generate_summary(self) -> None:
        """生成性能总结"""
        # 收集关键指标
        all_times = []
        all_ratios = []
        
        for test_data in self.results["multimodal_tests"].values():
            all_times.append(test_data["avg_pipeline_time"])
        
        for ratio_data in self.results["comparative_analysis"]["multimodal_ratios"].values():
            all_ratios.append(ratio_data["ratio_vs_single_text"])
        
        # 计算统计
        if all_times:
            avg_time = statistics.mean(all_times)
            max_time = max(all_times)
            min_time = min(all_times)
        else:
            avg_time = max_time = min_time = 0
        
        if all_ratios:
            avg_ratio = statistics.mean(all_ratios)
            max_ratio = max(all_ratios)
            min_ratio = min(all_ratios)
        else:
            avg_ratio = max_ratio = min_ratio = 0
        
        # 评估性能目标
        # 目标1：平均处理时间 < 2秒
        time_target_met = avg_time < 2.0
        
        # 目标2：多模态处理耗时 < 单模态1.5倍（中期目标）
        ratio_target_met = avg_ratio < 1.5
        
        # 目标3：最大处理时间 < 5秒
        max_time_target_met = max_time < 5.0
        
        # 总体评估
        overall_passed = time_target_met and ratio_target_met and max_time_target_met
        
        self.results["summary"] = {
            "statistics": {
                "avg_processing_time": avg_time,
                "max_processing_time": max_time,
                "min_processing_time": min_time,
                "avg_ratio_vs_single": avg_ratio,
                "max_ratio_vs_single": max_ratio,
                "min_ratio_vs_single": min_ratio
            },
            "targets": {
                "avg_time_under_2s": {"target": 2.0, "actual": avg_time, "met": time_target_met},
                "avg_ratio_under_1.5x": {"target": 1.5, "actual": avg_ratio, "met": ratio_target_met},
                "max_time_under_5s": {"target": 5.0, "actual": max_time, "met": max_time_target_met}
            },
            "overall_assessment": {
                "passed": overall_passed,
                "performance_class": self._classify_performance(avg_ratio),
                "recommendations": self._generate_recommendations(avg_time, avg_ratio, max_time)
            },
            "test_summary": {
                "total_single_modality_tests": len(self.results["single_modality_tests"]),
                "total_multimodal_tests": len(self.results["multimodal_tests"]),
                "total_iterations": self.num_iterations
            }
        }
    
    def _generate_recommendations(self, avg_time: float, avg_ratio: float, max_time: float) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if avg_time > 2.0:
            recommendations.append("优化处理流程以减少平均处理时间")
        
        if avg_ratio > 1.5:
            recommendations.append("改进多模态并行处理以减少与单模态的时间比率")
        
        if max_time > 5.0:
            recommendations.append("优化复杂场景处理以减少最大处理时间")
        
        if avg_time < 1.0 and avg_ratio < 1.2:
            recommendations.append("性能优秀，考虑进一步优化内存使用")
        elif avg_time < 1.5 and avg_ratio < 1.3:
            recommendations.append("性能良好，可进行微调优化")
        else:
            recommendations.append("需要重点关注性能优化")
        
        return recommendations
    
    def print_report(self) -> None:
        """打印性能报告"""
        summary = self.results["summary"]
        
        print("\n" + "=" * 80)
        print("性能基准测试报告")
        print("=" * 80)
        
        # 统计信息
        print("\n📊 统计信息:")
        stats = summary["statistics"]
        print(f"  平均处理时间: {stats['avg_processing_time']:.3f}秒")
        print(f"  最大处理时间: {stats['max_processing_time']:.3f}秒")
        print(f"  最小处理时间: {stats['min_processing_time']:.3f}秒")
        print(f"  平均与单模态比率: {stats['avg_ratio_vs_single']:.2f}x")
        print(f"  最大与单模态比率: {stats['max_ratio_vs_single']:.2f}x")
        
        # 目标评估
        print("\n🎯 目标评估:")
        targets = summary["targets"]
        for target_name, target_data in targets.items():
            status = "✓" if target_data["met"] else "✗"
            print(f"  {status} {target_name}: {target_data['actual']:.3f} / {target_data['target']:.1f}")
        
        # 总体评估
        print("\n📈 总体评估:")
        assessment = summary["overall_assessment"]
        status = "通过" if assessment["passed"] else "未通过"
        print(f"  测试结果: {status}")
        print(f"  性能等级: {assessment['performance_class']}")
        
        if assessment["recommendations"]:
            print("\n💡 改进建议:")
            for i, rec in enumerate(assessment["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        # 详细结果
        print("\n🔍 详细结果:")
        print("  单模态测试:", len(self.results["single_modality_tests"]))
        print("  多模态测试:", len(self.results["multimodal_tests"]))
        print("  迭代次数:", self.num_iterations)
        
        # 比较分析摘要
        print("\n📐 比较分析摘要:")
        ratios = self.results["comparative_analysis"]["multimodal_ratios"]
        for test_name, ratio_data in ratios.items():
            print(f"  {test_name}: {ratio_data['ratio_vs_single_text']:.2f}x ({ratio_data['performance_class']})")


def main():
    """主函数"""
    print("多模态性能基准测试启动...")
    
    # 创建基准测试实例
    benchmark = PerformanceBenchmark(num_iterations=5)  # 使用较少迭代以加快测试
    
    # 运行基准测试
    results = benchmark.run_benchmark()
    
    # 打印报告
    benchmark.print_report()
    
    # 保存结果
    output_file = "multimodal_performance_benchmark.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n性能测试结果已保存到: {output_file}")
    
    # 返回退出码
    summary = results["summary"]["overall_assessment"]
    return 0 if summary.get("passed", False) else 1


if __name__ == "__main__":
    sys.exit(main())