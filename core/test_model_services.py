"""
模型服务测试脚本
用于验证模型服务管理器的功能
"""
import os
import sys
import time
import requests
import json
import logging
from concurrent.futures import ThreadPoolExecutor

# 添加根目录到sys.path以便绝对导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_service_manager import model_service_manager
from core.model_ports_config import get_all_model_ports

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelServiceTester")

class ModelServiceTester:
    """模型服务测试类"""
    
    def __init__(self):
        self.model_ports = get_all_model_ports()
        
    def test_model_service(self, model_id: str) -> dict:
        """测试指定模型的服务"""
        port = self.model_ports.get(model_id)
        if not port:
            return {
                "model_id": model_id,
                "status": "error",
                "message": f"模型 {model_id} 未配置端口"
            }
        
        base_url = f"http://localhost:{port}"
        health_endpoint = f"{base_url}/{model_id}/health"
        
        try:
            # 测试健康检查端点
            start_time = time.time()
            response = requests.get(health_endpoint, timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                return {
                    "model_id": model_id,
                    "status": "success",
                    "port": port,
                    "response_time": round(response_time, 3),
                    "health_data": health_data
                }
            else:
                return {
                    "model_id": model_id,
                    "status": "error",
                    "port": port,
                    "status_code": response.status_code,
                    "message": f"健康检查失败，状态码: {response.status_code}"
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "model_id": model_id,
                "status": "error",
                "port": port,
                "message": "连接失败，服务可能未启动"
            }
        except requests.exceptions.Timeout:
            return {
                "model_id": model_id,
                "status": "error",
                "port": port,
                "message": "请求超时"
            }
        except Exception as e:
            return {
                "model_id": model_id,
                "status": "error",
                "port": port,
                "message": str(e)
            }
    
    def test_all_model_services(self, max_workers: int = 5) -> dict:
        """测试所有模型服务"""
        results = {}
        
        # 使用线程池并发测试
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有测试任务
            future_to_model = {
                executor.submit(self.test_model_service, model_id): model_id 
                for model_id in self.model_ports.keys()
            }
            
            # 获取测试结果
            for future in future_to_model:
                model_id = future_to_model[future]
                try:
                    results[model_id] = future.result()
                except Exception as e:
                    results[model_id] = {
                        "model_id": model_id,
                        "status": "error",
                        "message": str(e)
                    }
        
        return results
    
    def print_test_summary(self, results: dict):
        """打印测试结果摘要"""
        total = len(results)
        success = sum(1 for model_id in results if results[model_id]["status"] == "success")
        error = total - success
        
        logger.info(f"=== 模型服务测试摘要 ===")
        logger.info(f"总模型数: {total}")
        logger.info(f"成功: {success}")
        logger.info(f"失败: {error}")
        logger.info(f"=====================")
        
        if error > 0:
            logger.warning("失败的模型服务:")
            for model_id in results:
                if results[model_id]["status"] == "error":
                    logger.warning(f"  - {model_id}: {results[model_id].get('message', '未知错误')}")

    def save_results_to_file(self, results: dict, filename: str = "model_service_test_results.json"):
        """保存测试结果到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"测试结果已保存到 {filename}")
        except Exception as e:
            logger.error(f"保存测试结果失败: {str(e)}")

    def generate_service_report(self, results: dict, filename: str = "model_service_report.md"):
        """生成模型服务报告"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# 模型服务状态报告\n\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## 汇总统计\n")
                f.write(f"- 总模型数: {len(results)}\n")
                f.write(f"- 成功: {sum(1 for model_id in results if results[model_id]["status"] == "success")}\n")
                f.write(f"- 失败: {sum(1 for model_id in results if results[model_id]["status"] == "error")}\n\n")
                
                f.write("## 详细状态\n")
                f.write("| 模型ID | 状态 | 端口 | 响应时间(秒) | 消息 |\n")
                f.write("|-------|------|------|------------|------|\n")
                
                for model_id in sorted(results.keys()):
                    result = results[model_id]
                    status = "✅ 成功" if result["status"] == "success" else "❌ 失败"
                    port = result.get("port", "N/A")
                    response_time = result.get("response_time", "N/A")
                    message = result.get("message", "")
                    
                    f.write(f"| {model_id} | {status} | {port} | {response_time} | {message} |\n")
            
            logger.info(f"模型服务报告已生成: {filename}")
        except Exception as e:
            logger.error(f"生成模型服务报告失败: {str(e)}")

if __name__ == "__main__":
    logger.info("开始测试模型服务...")
    
    # 创建测试器实例
    tester = ModelServiceTester()
    
    # 等待服务启动
    logger.info("等待5秒让服务完全启动...")
    time.sleep(5)
    
    # 测试所有模型服务
    results = tester.test_all_model_services()
    
    # 打印测试摘要
    tester.print_test_summary(results)
    
    # 保存测试结果
    tester.save_results_to_file(results)
    
    # 生成服务报告
    tester.generate_service_report(results)
    
    logger.info("模型服务测试完成")