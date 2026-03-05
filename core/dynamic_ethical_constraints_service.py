#!/usr/bin/env python3
"""
动态伦理约束服务 - Dynamic Ethical Constraints Service

在8020端口提供动态伦理约束评估、场景化规则管理和人类反馈集成功能。
支持实时违规拦截、规则优化和跨场景伦理违规率监控。

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import uvicorn
import logging
import json
import time
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Body, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

# 导入动态伦理约束系统
from core.dynamic_ethical_constraints import (
    DynamicEthicalConstraints,
    get_ethical_constraints,
    ScenarioType,
    EthicalPrinciple,
    ViolationSeverity,
    HumanFeedbackType
)
from core.error_handling import error_handler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 服务配置
DYNAMIC_ETHICAL_CONSTRAINTS_SERVICE_PORT = 8020
SERVICE_HOST = os.environ.get("DYNAMIC_ETHICAL_CONSTRAINTS_HOST", "127.0.0.1")

# 数据模型
class EthicalEvaluationRequest(BaseModel):
    """伦理评估请求"""
    action: str = Field(..., description="要评估的行动或文本")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    require_correction: bool = Field(default=True, description="是否需要修正建议")
    language: str = Field(default="en", description="语言代码")

class HumanFeedbackRequest(BaseModel):
    """人类反馈请求"""
    action: str = Field(..., description="原始行动或文本")
    evaluation_result: Dict[str, Any] = Field(..., description="原始评估结果")
    feedback_type: str = Field(..., description="反馈类型 (approval, rejection, correction, suggestion, clarification)")
    feedback_details: Dict[str, Any] = Field(default_factory=dict, description="反馈详细信息")

class AddScenarioRuleRequest(BaseModel):
    """添加场景规则请求"""
    scenario_type: str = Field(..., description="场景类型")
    rule_name: str = Field(..., description="规则名称")
    rule_description: str = Field(..., description="规则描述")
    principle: str = Field(..., description="伦理原则")
    severity: str = Field(default="medium", description="违规严重程度")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    condition: Optional[str] = Field(default=None, description="匹配条件")
    action: str = Field(default="block", description="违规动作")
    weight: float = Field(default=1.0, description="规则权重")

class ScenarioDetectionRequest(BaseModel):
    """场景检测请求"""
    text: str = Field(..., description="输入文本")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")

class ExportRulesRequest(BaseModel):
    """导出规则请求"""
    format_type: str = Field(default="json", description="导出格式 (json, csv)")

class ImportRulesRequest(BaseModel):
    """导入规则请求"""
    import_data: str = Field(..., description="导入数据")
    format_type: str = Field(default="json", description="导入格式 (json, csv)")
    merge_strategy: str = Field(default="merge", description="合并策略 (merge, replace, update)")

class BatchEvaluationRequest(BaseModel):
    """批量评估请求"""
    actions: List[str] = Field(..., description="要评估的行动列表")
    contexts: Optional[List[Dict[str, Any]]] = Field(default=None, description="上下文信息列表")
    require_corrections: bool = Field(default=True, description="是否需要修正建议")

class DynamicEthicalConstraintsService:
    """动态伦理约束服务主类"""
    
    def __init__(self, port: int = DYNAMIC_ETHICAL_CONSTRAINTS_SERVICE_PORT):
        self.port = port
        self.app = FastAPI(
            title="Dynamic Ethical Constraints Service",
            version="1.0.0",
            description="AGI动态伦理约束服务，提供场景化伦理评估、规则管理和人类反馈集成功能",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # 核心组件
        self.ethical_constraints = get_ethical_constraints()
        
        # 服务状态
        self.is_running = False
        self.start_time = None
        self.request_count = 0
        
        # HTTP客户端
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # 配置CORS
        self._configure_cors()
        
        # 注册路由
        self._register_routes()
        
        logger.info(f"动态伦理约束服务初始化完成，端口: {self.port}")
    
    def _configure_cors(self):
        """配置CORS"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 生产环境应限制
            allow_credentials=False,  # 安全考虑，开发环境设为False
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.get("/")
        async def root():
            """根路径 - 服务信息"""
            return {
                "service": "Dynamic Ethical Constraints Service",
                "version": "1.0.0",
                "status": "running" if self.is_running else "stopped",
                "uptime": time.time() - self.start_time if self.start_time else 0,
                "request_count": self.request_count,
                "documentation": "/docs",
                "endpoints": [
                    {"path": "/evaluate", "method": "POST", "description": "伦理评估"},
                    {"path": "/batch-evaluate", "method": "POST", "description": "批量伦理评估"},
                    {"path": "/detect-scenario", "method": "POST", "description": "场景检测"},
                    {"path": "/add-feedback", "method": "POST", "description": "添加人类反馈"},
                    {"path": "/add-rule", "method": "POST", "description": "添加场景规则"},
                    {"path": "/system-status", "method": "GET", "description": "系统状态"},
                    {"path": "/violation-rate", "method": "GET", "description": "违规率统计"},
                    {"path": "/export-rules", "method": "POST", "description": "导出规则"},
                    {"path": "/import-rules", "method": "POST", "description": "导入规则"},
                    {"path": "/scenarios", "method": "GET", "description": "获取场景列表"},
                    {"path": "/principles", "method": "GET", "description": "获取伦理原则列表"},
                    {"path": "/severity-levels", "method": "GET", "description": "获取严重程度列表"},
                    {"path": "/feedback-types", "method": "GET", "description": "获取反馈类型列表"},
                ]
            }
        
        @self.app.post("/evaluate")
        async def evaluate_action(request: EthicalEvaluationRequest):
            """评估单个行动"""
            self.request_count += 1
            
            try:
                result = self.ethical_constraints.evaluate_action(
                    action=request.action,
                    context=request.context,
                    require_correction=request.require_correction
                )
                
                # 添加服务元数据
                result['service_metadata'] = {
                    'service_version': '1.0.0',
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'request_id': f"req_{self.request_count:08d}",
                }
                
                return JSONResponse(content=result)
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "评估行动时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/batch-evaluate")
        async def batch_evaluate_actions(request: BatchEvaluationRequest):
            """批量评估行动"""
            self.request_count += 1
            
            try:
                results = []
                contexts = request.contexts or [{}] * len(request.actions)
                
                if len(contexts) != len(request.actions):
                    raise HTTPException(
                        status_code=400, 
                        detail="Contexts list length must match actions list length"
                    )
                
                for i, action in enumerate(request.actions):
                    context = contexts[i]
                    
                    result = self.ethical_constraints.evaluate_action(
                        action=action,
                        context=context,
                        require_correction=request.require_corrections
                    )
                    
                    results.append(result)
                
                # 计算批量统计
                ethical_count = sum(1 for r in results if r.get('is_ethical', False))
                violation_count = len(results) - ethical_count
                
                batch_result = {
                    'results': results,
                    'batch_statistics': {
                        'total_actions': len(results),
                        'ethical_actions': ethical_count,
                        'violation_actions': violation_count,
                        'ethical_rate': ethical_count / len(results) if results else 0,
                        'violation_rate': violation_count / len(results) if results else 0,
                    },
                    'service_metadata': {
                        'service_version': '1.0.0',
                        'evaluation_timestamp': datetime.now().isoformat(),
                        'request_id': f"batch_{self.request_count:08d}",
                    }
                }
                
                return JSONResponse(content=batch_result)
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "批量评估时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/detect-scenario")
        async def detect_scenario(request: ScenarioDetectionRequest):
            """检测场景类型"""
            self.request_count += 1
            
            try:
                scenarios = self.ethical_constraints.detect_scenario(
                    text=request.text,
                    context=request.context
                )
                
                # 获取场景详细信息
                scenario_details = []
                for scenario in scenarios:
                    scenario_details.append({
                        'scenario_type': scenario,
                        'display_name': self._get_scenario_display_name(scenario),
                        'description': self._get_scenario_description(scenario),
                    })
                
                result = {
                    'text': request.text[:100] + "..." if len(request.text) > 100 else request.text,
                    'detected_scenarios': scenarios,
                    'scenario_details': scenario_details,
                    'primary_scenario': scenarios[0] if scenarios else None,
                    'service_metadata': {
                        'detection_timestamp': datetime.now().isoformat(),
                        'request_id': f"scenario_{self.request_count:08d}",
                    }
                }
                
                return JSONResponse(content=result)
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "检测场景时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/add-feedback")
        async def add_human_feedback(request: HumanFeedbackRequest):
            """添加人类反馈"""
            self.request_count += 1
            
            try:
                # 验证反馈类型
                if request.feedback_type not in [ft.value for ft in HumanFeedbackType]:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid feedback type. Valid types: {[ft.value for ft in HumanFeedbackType]}"
                    )
                
                self.ethical_constraints.add_human_feedback(
                    action=request.action,
                    evaluation_result=request.evaluation_result,
                    feedback_type=request.feedback_type,
                    feedback_details=request.feedback_details
                )
                
                result = {
                    'success': True,
                    'message': 'Human feedback added successfully',
                    'feedback_type': request.feedback_type,
                    'action_preview': request.action[:50] + "..." if len(request.action) > 50 else request.action,
                    'service_metadata': {
                        'feedback_timestamp': datetime.now().isoformat(),
                        'request_id': f"feedback_{self.request_count:08d}",
                    }
                }
                
                return JSONResponse(content=result)
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "添加人类反馈时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/add-rule")
        async def add_scenario_rule(request: AddScenarioRuleRequest):
            """添加场景规则"""
            self.request_count += 1
            
            try:
                # 构建规则数据
                rule_data = {
                    'name': request.rule_name,
                    'description': request.rule_description,
                    'principle': request.principle,
                    'severity': request.severity,
                    'keywords': request.keywords,
                    'condition': request.condition,
                    'action': request.action,
                    'weight': request.weight,
                    'confidence': 0.8,  # 默认置信度
                }
                
                rule_id = self.ethical_constraints.add_scenario_rule(
                    scenario_type=request.scenario_type,
                    rule_data=rule_data
                )
                
                result = {
                    'success': True,
                    'rule_id': rule_id,
                    'scenario_type': request.scenario_type,
                    'rule_name': request.rule_name,
                    'message': f"Rule '{request.rule_name}' added successfully with ID: {rule_id}",
                    'service_metadata': {
                        'creation_timestamp': datetime.now().isoformat(),
                        'request_id': f"rule_{self.request_count:08d}",
                    }
                }
                
                return JSONResponse(content=result)
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "添加场景规则时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/system-status")
        async def get_system_status():
            """获取系统状态"""
            self.request_count += 1
            
            try:
                status = self.ethical_constraints.get_system_status()
                
                # 添加服务状态
                status['service_status'] = {
                    'is_running': self.is_running,
                    'uptime_seconds': time.time() - self.start_time if self.start_time else 0,
                    'total_requests': self.request_count,
                    'port': self.port,
                    'host': SERVICE_HOST,
                }
                
                return JSONResponse(content=status)
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "获取系统状态时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/violation-rate")
        async def get_violation_rate(time_window_hours: int = Query(24, description="时间窗口（小时）")):
            """获取违规率统计"""
            self.request_count += 1
            
            try:
                if time_window_hours <= 0 or time_window_hours > 720:  # 限制最大30天
                    raise HTTPException(
                        status_code=400, 
                        detail="Time window must be between 1 and 720 hours"
                    )
                
                violation_rate = self.ethical_constraints.get_violation_rate(time_window_hours)
                
                return JSONResponse(content=violation_rate)
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "获取违规率时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/export-rules")
        async def export_rules(request: ExportRulesRequest):
            """导出规则库"""
            self.request_count += 1
            
            try:
                if request.format_type not in ['json', 'csv']:
                    raise HTTPException(
                        status_code=400, 
                        detail="Invalid format type. Valid types: json, csv"
                    )
                
                exported_data = self.ethical_constraints.export_rules(request.format_type)
                
                result = {
                    'success': True,
                    'format_type': request.format_type,
                    'data': exported_data,
                    'data_length': len(exported_data),
                    'message': f"Rules exported successfully in {request.format_type} format",
                    'service_metadata': {
                        'export_timestamp': datetime.now().isoformat(),
                        'request_id': f"export_{self.request_count:08d}",
                    }
                }
                
                return JSONResponse(content=result)
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "导出规则时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/import-rules")
        async def import_rules(request: ImportRulesRequest):
            """导入规则库"""
            self.request_count += 1
            
            try:
                if request.format_type not in ['json', 'csv']:
                    raise HTTPException(
                        status_code=400, 
                        detail="Invalid format type. Valid types: json, csv"
                    )
                
                if request.merge_strategy not in ['merge', 'replace', 'update']:
                    raise HTTPException(
                        status_code=400, 
                        detail="Invalid merge strategy. Valid strategies: merge, replace, update"
                    )
                
                import_result = self.ethical_constraints.import_rules(
                    import_data=request.import_data,
                    format_type=request.format_type,
                    merge_strategy=request.merge_strategy
                )
                
                return JSONResponse(content=import_result)
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "导入规则时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/scenarios")
        async def get_scenarios():
            """获取场景列表"""
            self.request_count += 1
            
            try:
                scenarios = []
                for scenario_type in ScenarioType:
                    scenario_data = {
                        'scenario_type': scenario_type.value,
                        'display_name': self._get_scenario_display_name(scenario_type.value),
                        'description': self._get_scenario_description(scenario_type.value),
                    }
                    scenarios.append(scenario_data)
                
                return JSONResponse(content={'scenarios': scenarios})
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "获取场景列表时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/principles")
        async def get_principles():
            """获取伦理原则列表"""
            self.request_count += 1
            
            try:
                principles = []
                for principle in EthicalPrinciple:
                    principle_data = {
                        'principle': principle.value,
                        'display_name': self._get_principle_display_name(principle.value),
                        'description': self._get_principle_description(principle.value),
                    }
                    principles.append(principle_data)
                
                return JSONResponse(content={'principles': principles})
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "获取伦理原则列表时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/severity-levels")
        async def get_severity_levels():
            """获取严重程度列表"""
            self.request_count += 1
            
            try:
                severity_levels = []
                for severity in ViolationSeverity:
                    severity_data = {
                        'severity': severity.value,
                        'display_name': self._get_severity_display_name(severity.value),
                        'description': self._get_severity_description(severity.value),
                    }
                    severity_levels.append(severity_data)
                
                return JSONResponse(content={'severity_levels': severity_levels})
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "获取严重程度列表时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/feedback-types")
        async def get_feedback_types():
            """获取反馈类型列表"""
            self.request_count += 1
            
            try:
                feedback_types = []
                for feedback_type in HumanFeedbackType:
                    feedback_data = {
                        'feedback_type': feedback_type.value,
                        'display_name': self._get_feedback_type_display_name(feedback_type.value),
                        'description': self._get_feedback_type_description(feedback_type.value),
                    }
                    feedback_types.append(feedback_data)
                
                return JSONResponse(content={'feedback_types': feedback_types})
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraintsService", "获取反馈类型列表时出错")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "service": "Dynamic Ethical Constraints Service",
                "timestamp": datetime.now().isoformat(),
                "uptime": time.time() - self.start_time if self.start_time else 0,
            }
        
        @self.app.on_event("startup")
        async def startup_event():
            """服务启动事件"""
            self.is_running = True
            self.start_time = time.time()
            logger.info(f"动态伦理约束服务启动成功，监听 {SERVICE_HOST}:{self.port}")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """服务关闭事件"""
            self.is_running = False
            logger.info("动态伦理约束服务关闭")
    
    def _get_scenario_display_name(self, scenario_type: str) -> str:
        """获取场景显示名称"""
        display_names = {
            ScenarioType.GENERAL.value: "通用场景",
            ScenarioType.MEDICAL.value: "医疗场景", 
            ScenarioType.FINANCE.value: "金融场景",
            ScenarioType.LEGAL.value: "法律场景",
            ScenarioType.EDUCATION.value: "教育场景",
            ScenarioType.INDUSTRIAL.value: "工业场景",
            ScenarioType.RESEARCH.value: "科研场景",
            ScenarioType.CREATIVE.value: "创意场景",
            ScenarioType.PERSONAL.value: "个人场景",
            ScenarioType.ROBOTICS.value: "机器人场景",
        }
        return display_names.get(scenario_type, scenario_type)
    
    def _get_scenario_description(self, scenario_type: str) -> str:
        """获取场景描述"""
        descriptions = {
            ScenarioType.GENERAL.value: "通用对话和交互场景",
            ScenarioType.MEDICAL.value: "医疗健康相关的对话和咨询",
            ScenarioType.FINANCE.value: "金融投资相关的对话和咨询", 
            ScenarioType.LEGAL.value: "法律相关的对话和咨询",
            ScenarioType.EDUCATION.value: "教育学习相关的对话和指导",
            ScenarioType.INDUSTRIAL.value: "工业生产和安全相关的对话",
            ScenarioType.RESEARCH.value: "科学研究和实验相关的对话",
            ScenarioType.CREATIVE.value: "创意写作和艺术相关的对话",
            ScenarioType.PERSONAL.value: "个人生活和隐私相关的对话",
            ScenarioType.ROBOTICS.value: "机器人控制和操作相关的对话",
        }
        return descriptions.get(scenario_type, "Unknown scenario")
    
    def _get_principle_display_name(self, principle: str) -> str:
        """获取伦理原则显示名称"""
        display_names = {
            EthicalPrinciple.BENEFICENCE.value: "仁慈原则",
            EthicalPrinciple.NON_MALEFICENCE.value: "非恶意原则",
            EthicalPrinciple.AUTONOMY.value: "自主原则",
            EthicalPrinciple.JUSTICE.value: "公正原则",
            EthicalPrinciple.TRUTHFULNESS.value: "诚实原则",
            EthicalPrinciple.CONFIDENTIALITY.value: "保密原则",
            EthicalPrinciple.ACCOUNTABILITY.value: "问责原则",
            EthicalPrinciple.TRANSPARENCY.value: "透明原则",
        }
        return display_names.get(principle, principle)
    
    def _get_principle_description(self, principle: str) -> str:
        """获取伦理原则描述"""
        descriptions = {
            EthicalPrinciple.BENEFICENCE.value: "行善、促进福祉",
            EthicalPrinciple.NON_MALEFICENCE.value: "不伤害、避免伤害",
            EthicalPrinciple.AUTONOMY.value: "尊重个体自主权和选择权",
            EthicalPrinciple.JUSTICE.value: "公平正义、平等对待",
            EthicalPrinciple.TRUTHFULNESS.value: "说真话、不欺骗",
            EthicalPrinciple.CONFIDENTIALITY.value: "保护隐私和机密信息",
            EthicalPrinciple.ACCOUNTABILITY.value: "承担责任、可问责",
            EthicalPrinciple.TRANSPARENCY.value: "操作透明、可解释",
        }
        return descriptions.get(principle, "Unknown principle")
    
    def _get_severity_display_name(self, severity: str) -> str:
        """获取严重程度显示名称"""
        display_names = {
            ViolationSeverity.LOW.value: "低",
            ViolationSeverity.MEDIUM.value: "中",
            ViolationSeverity.HIGH.value: "高",
            ViolationSeverity.CRITICAL.value: "关键",
        }
        return display_names.get(severity, severity)
    
    def _get_severity_description(self, severity: str) -> str:
        """获取严重程度描述"""
        descriptions = {
            ViolationSeverity.LOW.value: "轻微违规，可警告",
            ViolationSeverity.MEDIUM.value: "中等违规，需要纠正",
            ViolationSeverity.HIGH.value: "严重违规，必须阻止",
            ViolationSeverity.CRITICAL.value: "极度危险，立即终止",
        }
        return descriptions.get(severity, "Unknown severity")
    
    def _get_feedback_type_display_name(self, feedback_type: str) -> str:
        """获取反馈类型显示名称"""
        display_names = {
            HumanFeedbackType.APPROVAL.value: "批准",
            HumanFeedbackType.REJECTION.value: "拒绝",
            HumanFeedbackType.CORRECTION.value: "修正",
            HumanFeedbackType.SUGGESTION.value: "建议",
            HumanFeedbackType.CLARIFICATION.value: "澄清",
        }
        return display_names.get(feedback_type, feedback_type)
    
    def _get_feedback_type_description(self, feedback_type: str) -> str:
        """获取反馈类型描述"""
        descriptions = {
            HumanFeedbackType.APPROVAL.value: "操作符合伦理",
            HumanFeedbackType.REJECTION.value: "操作违反伦理",
            HumanFeedbackType.CORRECTION.value: "部分符合，需要调整",
            HumanFeedbackType.SUGGESTION.value: "改进建议",
            HumanFeedbackType.CLARIFICATION.value: "需要更多上下文",
        }
        return descriptions.get(feedback_type, "Unknown feedback type")
    
    async def run(self):
        """运行服务"""
        config = uvicorn.Config(
            self.app,
            host=SERVICE_HOST,
            port=self.port,
            log_level="info",
            access_log=True,
        )
        server = uvicorn.Server(config)
        await server.serve()


def main():
    """主函数"""
    service = DynamicEthicalConstraintsService()
    
    # 运行服务
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        logger.info("服务被用户中断")
    except Exception as e:
        error_handler.handle_error(e, "DynamicEthicalConstraintsService", "服务运行失败")
        raise


if __name__ == "__main__":
    main()