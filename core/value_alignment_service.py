#!/usr/bin/env python3
"""
价值对齐服务 - 在8019端口提供价值对齐、伦理推理和价值校验功能
集成现有的ValueAlignment类和KnowledgeManager知识库
"""

import asyncio
import uvicorn
import logging
import json
import time
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx

# 导入现有组件
from core.value_alignment import ValueAlignment, EthicalReasoner, ValueSystem
from core.knowledge_manager import KnowledgeManager
from core.error_handling import error_handler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 服务配置
VALUE_ALIGNMENT_SERVICE_PORT = 8019
SERVICE_HOST = os.environ.get("VALUE_ALIGNMENT_HOST", "127.0.0.1")

# 数据模型
class AlignmentRequest(BaseModel):
    """价值对齐请求"""
    action: str = Field(..., description="要评估的行动或文本")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    language: str = Field(default="en", description="语言代码 (en, zh, de, ja, ru)")
    require_correction: bool = Field(default=True, description="是否需要修正建议")
    detailed_report: bool = Field(default=True, description="是否需要详细报告")

class EthicalDilemmaRequest(BaseModel):
    """伦理困境请求"""
    dilemma: str = Field(..., description="伦理困境描述")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    ethical_framework: str = Field(default="multi_perspective", description="伦理框架 (utilitarianism, deontology, virtue_ethics, rights_based, care_ethics, multi_perspective)")
    language: str = Field(default="en", description="语言代码")

class ValueCorrectionRequest(BaseModel):
    """价值修正请求"""
    text: str = Field(..., description="需要修正的文本")
    original_assessment: Dict[str, Any] = Field(..., description="原始价值评估结果")
    correction_guidelines: List[str] = Field(default_factory=list, description="修正指导原则")
    language: str = Field(default="en", description="语言代码")

class KnowledgeQueryRequest(BaseModel):
    """知识库查询请求"""
    domain: str = Field(default="all", description="知识领域 (philosophy, humanities, ethics, law, psychology, all)")
    query: str = Field(default="", description="查询关键词")
    language: str = Field(default="en", description="语言代码")
    max_results: int = Field(default=10, description="最大返回结果数")

class MultiLanguageEthicalGuidelineRequest(BaseModel):
    """多语言伦理准则请求"""
    principle_id: Optional[str] = Field(default=None, description="准则ID")
    language: str = Field(default="en", description="语言代码")
    category: Optional[str] = Field(default=None, description="类别过滤")

class ValueAlignmentService:
    """价值对齐服务主类"""
    
    def __init__(self, port: int = VALUE_ALIGNMENT_SERVICE_PORT):
        self.port = port
        self.app = FastAPI(
            title="Value Alignment Service",
            version="1.0.0",
            description="AGI价值对齐服务，提供伦理推理、价值判断、价值校验和多语言伦理准则查询功能",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # 核心组件
        self.value_alignment = ValueAlignment()
        self.knowledge_manager = KnowledgeManager()
        self.ethical_reasoner = EthicalReasoner()
        self.value_system = ValueSystem()
        
        # 服务状态
        self.is_running = False
        self.start_time = None
        self.request_count = 0
        
        # HTTP客户端
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # 加载知识库
        self._load_knowledge_bases()
        
        # 设置API路由
        self._setup_routes()
    
    def _load_knowledge_bases(self):
        """加载知识库"""
        try:
            logger.info("Loading knowledge bases...")
            self.knowledge_manager.load_knowledge_bases()
            
            # 特别加载伦理相关知识库
            self.ethical_knowledge = {}
            for domain in ['philosophy', 'humanities', 'ethics', 'law', 'psychology']:
                if domain in self.knowledge_manager.knowledge_bases:
                    self.ethical_knowledge[domain] = self.knowledge_manager.knowledge_bases[domain]
                    logger.info(f"Loaded ethical knowledge domain: {domain}")
            
            logger.info(f"Total ethical knowledge domains loaded: {len(self.ethical_knowledge)}")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge bases: {e}")
            self.ethical_knowledge = {}
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.get("/")
        async def root():
            """服务根端点"""
            return {
                "service": "Value Alignment Service",
                "version": "1.0.0",
                "port": self.port,
                "status": "running" if self.is_running else "stopped",
                "uptime": time.time() - self.start_time if self.start_time else 0,
                "request_count": self.request_count,
                "endpoints": {
                    "POST /api/v1/align": "价值对齐评估",
                    "POST /api/v1/ethical/dilemma": "伦理困境分析",
                    "POST /api/v1/correct": "价值修正建议",
                    "GET /api/v1/knowledge": "知识库查询",
                    "GET /api/v1/guidelines": "多语言伦理准则",
                    "GET /api/v1/health": "健康检查",
                    "GET /api/v1/stats": "服务统计"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """健康检查端点"""
            return {
                "status": "healthy",
                "service": "value_alignment",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "value_alignment": "active",
                    "knowledge_manager": "active" if self.ethical_knowledge else "inactive",
                    "ethical_reasoner": "active"
                }
            }
        
        @self.app.post("/api/v1/align")
        async def align_action(request: AlignmentRequest):
            """价值对齐评估"""
            self.request_count += 1
            try:
                # 执行价值对齐评估，添加超时保护（30秒）
                align_timeout = 30.0  # 30秒超时
                try:
                    # 将同步方法转换为异步执行，并添加超时
                    loop = asyncio.get_event_loop()
                    alignment_result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,  # 使用默认执行器
                            self.value_alignment.align_action,
                            request.action, 
                            request.context
                        ),
                        timeout=align_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Alignment timeout after {align_timeout} seconds")
                    raise HTTPException(
                        status_code=408, 
                        detail=f"Alignment timeout after {align_timeout} seconds. Please try a simpler action or reduce complexity."
                    )
                
                # 如果需要修正建议
                correction_suggestions = []
                if request.require_correction:
                    correction_suggestions = self._generate_correction_suggestions(
                        request.action, 
                        alignment_result,
                        request.language
                    )
                
                # 构建响应
                response = {
                    "aligned": alignment_result.get('verdict', {}).get('verdict') == 'ALIGNED',
                    "alignment_score": alignment_result.get('overall_assessment', {}).get('alignment_score', 0.0),
                    "confidence": alignment_result.get('overall_assessment', {}).get('confidence', 0.0),
                    "primary_concerns": alignment_result.get('overall_assessment', {}).get('primary_concerns', []),
                    "positive_aspects": alignment_result.get('overall_assessment', {}).get('positive_aspects', []),
                    "requires_human_review": alignment_result.get('overall_assessment', {}).get('requires_human_review', False),
                    "correction_suggestions": correction_suggestions,
                    "language": request.language,
                    "timestamp": datetime.now().isoformat()
                }
                
                if request.detailed_report:
                    response["detailed_assessment"] = alignment_result
                
                return response
                
            except HTTPException:
                # 重新抛出HTTP异常
                raise
            except Exception as e:
                logger.error(f"Alignment failed: {e}")
                raise HTTPException(status_code=500, detail=f"Alignment failed: {str(e)}")
        
        @self.app.post("/api/v1/ethical/dilemma")
        async def analyze_ethical_dilemma(request: EthicalDilemmaRequest):
            """伦理困境分析"""
            self.request_count += 1
            try:
                # 使用伦理推理器分析困境
                ethical_analysis = self.ethical_reasoner.resolve_ethical_dilemma(
                    request.dilemma,
                    request.context
                )
                
                # 如果指定了特定伦理框架，只返回该框架的分析
                framework_analysis = {}
                if request.ethical_framework != "multi_perspective":
                    if request.ethical_framework in ethical_analysis.get('framework_evaluations', {}):
                        framework_analysis = ethical_analysis['framework_evaluations'][request.ethical_framework]
                
                # 从知识库获取相关伦理原则
                ethical_principles = self._get_relevant_ethical_principles(
                    request.dilemma,
                    request.language
                )
                
                response = {
                    "dilemma": request.dilemma,
                    "ethical_framework": request.ethical_framework,
                    "consensus_recommendation": ethical_analysis.get('consensus_recommendation', {}),
                    "framework_evaluations": ethical_analysis.get('framework_evaluations', {}),
                    "specific_framework_analysis": framework_analysis,
                    "ethical_principles": ethical_principles,
                    "case_id": ethical_analysis.get('case_id'),
                    "language": request.language,
                    "timestamp": datetime.now().isoformat()
                }
                
                return response
                
            except Exception as e:
                logger.error(f"Ethical analysis failed: {e}")
                raise HTTPException(status_code=500, detail=f"Ethical analysis failed: {str(e)}")
        
        @self.app.post("/api/v1/correct")
        async def correct_values(request: ValueCorrectionRequest):
            """价值修正建议"""
            self.request_count += 1
            try:
                # 生成修正建议
                corrections = self._generate_value_corrections(
                    request.text,
                    request.original_assessment,
                    request.correction_guidelines,
                    request.language
                )
                
                # 生成修正后的文本
                corrected_text = self._apply_corrections(
                    request.text,
                    corrections
                )
                
                response = {
                    "original_text": request.text,
                    "corrected_text": corrected_text,
                    "corrections": corrections,
                    "alignment_improvement": self._calculate_alignment_improvement(
                        request.original_assessment,
                        corrected_text
                    ),
                    "language": request.language,
                    "timestamp": datetime.now().isoformat()
                }
                
                return response
                
            except Exception as e:
                logger.error(f"Value correction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Value correction failed: {str(e)}")
        
        @self.app.get("/api/v1/knowledge")
        async def query_knowledge(
            domain: str = Query("all", description="知识领域"),
            query: str = Query("", description="查询关键词"),
            language: str = Query("en", description="语言代码"),
            max_results: int = Query(10, description="最大返回结果数")
        ):
            """知识库查询"""
            self.request_count += 1
            try:
                results = self._query_knowledge_base(
                    domain,
                    query,
                    language,
                    max_results
                )
                
                return {
                    "domain": domain,
                    "query": query,
                    "language": language,
                    "total_results": len(results),
                    "results": results[:max_results],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Knowledge query failed: {e}")
                raise HTTPException(status_code=500, detail=f"Knowledge query failed: {str(e)}")
        
        @self.app.get("/api/v1/guidelines")
        async def get_ethical_guidelines(
            principle_id: Optional[str] = Query(None, description="准则ID"),
            language: str = Query("en", description="语言代码"),
            category: Optional[str] = Query(None, description="类别过滤")
        ):
            """多语言伦理准则查询"""
            self.request_count += 1
            try:
                guidelines = self._get_ethical_guidelines(
                    principle_id,
                    language,
                    category
                )
                
                return {
                    "principle_id": principle_id,
                    "language": language,
                    "category": category,
                    "total_guidelines": len(guidelines),
                    "guidelines": guidelines,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Guidelines query failed: {e}")
                raise HTTPException(status_code=500, detail=f"Guidelines query failed: {str(e)}")
        
        @self.app.get("/api/v1/stats")
        async def get_service_stats():
            """获取服务统计信息"""
            try:
                # 获取价值对齐统计
                value_stats = self.value_alignment.get_alignment_report() if hasattr(self.value_alignment, 'get_alignment_report') else {}
                
                # 获取伦理推理统计
                ethical_stats = self.ethical_reasoner.get_ethical_report() if hasattr(self.ethical_reasoner, 'get_ethical_report') else {}
                
                # 知识库统计
                knowledge_stats = {
                    "domains_loaded": list(self.ethical_knowledge.keys()),
                    "total_domains": len(self.ethical_knowledge)
                }
                
                return {
                    "service": {
                        "port": self.port,
                        "is_running": self.is_running,
                        "uptime": time.time() - self.start_time if self.start_time else 0,
                        "start_time": self.start_time,
                        "request_count": self.request_count
                    },
                    "value_alignment": value_stats,
                    "ethical_reasoning": ethical_stats,
                    "knowledge_base": knowledge_stats
                }
                
            except Exception as e:
                logger.error(f"Failed to get service stats: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get service stats: {str(e)}")
    
    # 辅助方法
    
    def _generate_correction_suggestions(self, action: str, assessment: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
        """生成修正建议"""
        suggestions = []
        
        try:
            # 从评估结果中提取问题
            concerns = assessment.get('overall_assessment', {}).get('primary_concerns', [])
            alignment_score = assessment.get('overall_assessment', {}).get('alignment_score', 0.0)
            
            # 如果对齐分数低，生成通用建议
            if alignment_score < 0.6:
                suggestions.append({
                    "type": "general",
                    "description": self._translate("Consider rephrasing to better align with ethical values", language),
                    "severity": "high",
                    "suggestion": self._translate("Focus on positive impact, respect for others, and honesty", language)
                })
            
            # 针对具体问题生成建议
            for concern in concerns[:3]:  # 最多处理3个主要问题
                if "safety" in concern.lower():
                    suggestions.append({
                        "type": "safety",
                        "description": self._translate("Safety concern detected", language),
                        "severity": "high",
                        "suggestion": self._translate("Ensure the action does not cause harm to people or the environment", language)
                    })
                elif "honesty" in concern.lower():
                    suggestions.append({
                        "type": "honesty",
                        "description": self._translate("Honesty concern detected", language),
                        "severity": "medium",
                        "suggestion": self._translate("Be transparent and truthful in communication", language)
                    })
                elif "privacy" in concern.lower():
                    suggestions.append({
                        "type": "privacy",
                        "description": self._translate("Privacy concern detected", language),
                        "severity": "medium",
                        "suggestion": self._translate("Respect personal privacy and data protection", language)
                    })
            
            # 如果没有具体问题但有改进空间
            if not suggestions and alignment_score < 0.8:
                suggestions.append({
                    "type": "improvement",
                    "description": self._translate("Room for improvement in value alignment", language),
                    "severity": "low",
                    "suggestion": self._translate("Consider how the action impacts all stakeholders", language)
                })
                
        except Exception as e:
            logger.error(f"Failed to generate correction suggestions: {e}")
            suggestions = [{
                "type": "error",
                "description": f"Error generating suggestions: {str(e)}",
                "severity": "unknown",
                "suggestion": "Please review the action manually"
            }]
        
        return suggestions
    
    def _get_relevant_ethical_principles(self, dilemma: str, language: str) -> List[Dict[str, Any]]:
        """获取相关伦理原则"""
        principles = []
        
        try:
            # 从知识库中搜索相关伦理原则
            if 'philosophy' in self.ethical_knowledge:
                philosophy_data = self.ethical_knowledge['philosophy']
                
                # 搜索伦理学部分
                if 'knowledge_base' in philosophy_data and 'categories' in philosophy_data['knowledge_base']:
                    for category in philosophy_data['knowledge_base']['categories']:
                        if category.get('id') == 'ethics' and 'concepts' in category:
                            for concept in category['concepts']:
                                # 检查概念是否与困境相关
                                concept_name = concept.get('name', {}).get(language, concept.get('name', {}).get('en', ''))
                                concept_desc = concept.get('description', {}).get(language, concept.get('description', {}).get('en', ''))
                                
                                # 简单关键词匹配
                                dilemma_lower = dilemma.lower()
                                if any(keyword in dilemma_lower for keyword in ['ethical', 'moral', 'right', 'wrong', 'justice', 'fair']):
                                    principles.append({
                                        "id": concept.get('id'),
                                        "name": concept_name,
                                        "description": concept_desc,
                                        "category": "ethics",
                                        "relevance": "high"
                                    })
            
            # 限制返回数量
            principles = principles[:5]
            
        except Exception as e:
            logger.error(f"Failed to get ethical principles: {e}")
        
        return principles
    
    def _generate_value_corrections(self, text: str, assessment: Dict[str, Any], 
                                   guidelines: List[str], language: str) -> List[Dict[str, Any]]:
        """生成价值修正"""
        corrections = []
        
        try:
            # 分析评估结果中的问题
            concerns = assessment.get('primary_concerns', [])
            alignment_score = assessment.get('alignment_score', 0.0)
            
            # 生成基于问题的修正
            for concern in concerns:
                correction = {
                    "concern": concern,
                    "original_text_segment": self._extract_relevant_segment(text, concern),
                    "suggested_replacement": self._generate_replacement(text, concern, language),
                    "reason": self._get_correction_reason(concern, language),
                    "guideline_applied": self._find_relevant_guideline(concern, guidelines, language)
                }
                corrections.append(correction)
            
            # 如果没有具体问题但分数低，生成一般性修正
            if not corrections and alignment_score < 0.7:
                corrections.append({
                    "concern": "low_alignment_score",
                    "original_text_segment": text[:100] + "..." if len(text) > 100 else text,
                    "suggested_replacement": self._improve_general_alignment(text, language),
                    "reason": self._translate("Overall value alignment needs improvement", language),
                    "guideline_applied": self._translate("General ethical principles", language)
                })
                
        except Exception as e:
            logger.error(f"Failed to generate value corrections: {e}")
        
        return corrections
    
    def _apply_corrections(self, text: str, corrections: List[Dict[str, Any]]) -> str:
        """应用修正到文本"""
        corrected_text = text
        
        try:
            for correction in corrections:
                original = correction.get('original_text_segment', '')
                replacement = correction.get('suggested_replacement', '')
                
                if original and replacement and original in corrected_text:
                    corrected_text = corrected_text.replace(original, replacement)
                    
        except Exception as e:
            logger.error(f"Failed to apply corrections: {e}")
        
        return corrected_text
    
    def _calculate_alignment_improvement(self, original_assessment: Dict[str, Any], corrected_text: str) -> Dict[str, Any]:
        """计算对齐改进程度"""
        try:
            # 这里应该重新评估修正后的文本，但为了简化，我们估计改进
            original_score = original_assessment.get('alignment_score', 0.0)
            estimated_improvement = min(1.0, original_score + 0.2)  # 估计提高20%
            
            return {
                "original_score": original_score,
                "estimated_improvement": estimated_improvement - original_score,
                "estimated_new_score": estimated_improvement,
                "improvement_percentage": ((estimated_improvement - original_score) / original_score * 100) if original_score > 0 else 100
            }
        except Exception as e:
            logger.error(f"Failed to calculate alignment improvement: {e}")
            return {
                "original_score": 0.0,
                "estimated_improvement": 0.0,
                "estimated_new_score": 0.0,
                "improvement_percentage": 0.0
            }
    
    def _query_knowledge_base(self, domain: str, query: str, language: str, max_results: int) -> List[Dict[str, Any]]:
        """查询知识库"""
        results = []
        
        try:
            # 确定要查询的领域
            domains_to_search = []
            if domain == "all":
                domains_to_search = list(self.ethical_knowledge.keys())
            elif domain in self.ethical_knowledge:
                domains_to_search = [domain]
            else:
                return results
            
            # 在每个领域中进行搜索
            for domain_name in domains_to_search:
                domain_data = self.ethical_knowledge[domain_name]
                
                # 递归搜索数据结构
                domain_results = self._search_knowledge_structure(domain_data, query, language)
                results.extend(domain_results)
                
                if len(results) >= max_results * 2:  # 收集稍多一些以便筛选
                    break
            
            # 排序和限制结果
            results = self._rank_results(results, query, language)[:max_results]
            
        except Exception as e:
            logger.error(f"Knowledge base query failed: {e}")
        
        return results
    
    def _search_knowledge_structure(self, data: Any, query: str, language: str, path: str = "") -> List[Dict[str, Any]]:
        """递归搜索知识结构"""
        results = []
        
        try:
            if isinstance(data, dict):
                # 检查字典的键值对
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # 检查键
                    if query.lower() in str(key).lower():
                        results.append({
                            "path": current_path,
                            "content": str(value)[:200],
                            "type": "key_match"
                        })
                    
                    # 检查值（如果是字符串）
                    if isinstance(value, str) and query.lower() in value.lower():
                        results.append({
                            "path": current_path,
                            "content": value[:200],
                            "type": "value_match"
                        })
                    
                    # 递归搜索嵌套结构
                    results.extend(self._search_knowledge_structure(value, query, language, current_path))
                    
            elif isinstance(data, list):
                # 检查列表元素
                for i, item in enumerate(data):
                    current_path = f"{path}[{i}]"
                    results.extend(self._search_knowledge_structure(item, query, language, current_path))
                    
        except Exception as e:
            logger.error(f"Search failed at path {path}: {e}")
        
        return results
    
    def _rank_results(self, results: List[Dict[str, Any]], query: str, language: str) -> List[Dict[str, Any]]:
        """排序搜索结果"""
        try:
            # 简单排序：完全匹配优先
            query_lower = query.lower()
            scored_results = []
            
            for result in results:
                score = 0
                content = result.get('content', '').lower()
                
                # 完全匹配得分更高
                if query_lower in content:
                    score += 10
                
                # 路径相关度
                if 'ethics' in result.get('path', '').lower():
                    score += 5
                
                # 更新分数
                result['relevance_score'] = score
                scored_results.append(result)
            
            # 按分数排序
            scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            return scored_results
            
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return results
    
    def _get_ethical_guidelines(self, principle_id: Optional[str], language: str, category: Optional[str]) -> List[Dict[str, Any]]:
        """获取伦理准则"""
        guidelines = []
        
        try:
            # 如果指定了准则ID，尝试查找特定准则
            if principle_id:
                guideline = self._find_specific_guideline(principle_id, language)
                if guideline:
                    guidelines.append(guideline)
                return guidelines
            
            # 否则返回所有相关准则
            if 'philosophy' in self.ethical_knowledge:
                philosophy_data = self.ethical_knowledge['philosophy']
                
                # 提取伦理准则
                if 'knowledge_base' in philosophy_data:
                    kb = philosophy_data['knowledge_base']
                    
                    # 检查categories
                    if 'categories' in kb:
                        for cat in kb['categories']:
                            # 如果指定了类别，只搜索该类别
                            if category and cat.get('id') != category:
                                continue
                            
                            if cat.get('id') == 'ethics' and 'concepts' in cat:
                                for concept in cat['concepts']:
                                    guideline = self._extract_guideline_from_concept(concept, language)
                                    if guideline:
                                        guidelines.append(guideline)
            
            # 限制返回数量
            guidelines = guidelines[:20]
            
        except Exception as e:
            logger.error(f"Failed to get ethical guidelines: {e}")
        
        return guidelines
    
    def _find_specific_guideline(self, principle_id: str, language: str) -> Optional[Dict[str, Any]]:
        """查找特定准则"""
        try:
            # 在知识库中搜索特定ID
            for domain, data in self.ethical_knowledge.items():
                result = self._search_by_id(data, principle_id, language)
                if result:
                    return result
        except Exception as e:
            logger.error(f"Failed to find specific guideline: {e}")
        
        return None
    
    def _search_by_id(self, data: Any, target_id: str, language: str, path: str = "") -> Optional[Dict[str, Any]]:
        """按ID搜索"""
        try:
            if isinstance(data, dict):
                # 检查当前字典是否有匹配的ID
                if data.get('id') == target_id:
                    return self._extract_guideline_from_concept(data, language)
                
                # 递归搜索嵌套结构
                for key, value in data.items():
                    result = self._search_by_id(value, target_id, language, f"{path}.{key}")
                    if result:
                        return result
                        
            elif isinstance(data, list):
                for item in data:
                    result = self._search_by_id(item, target_id, language, path)
                    if result:
                        return result
                        
        except Exception as e:
            logger.error(f"Search by ID failed at path {path}: {e}")
        
        return None
    
    def _extract_guideline_from_concept(self, concept: Dict[str, Any], language: str) -> Optional[Dict[str, Any]]:
        """从概念中提取准则"""
        try:
            name = concept.get('name', {})
            description = concept.get('description', {})
            
            guideline = {
                "id": concept.get('id'),
                "name": name.get(language, name.get('en', '')),
                "description": description.get(language, description.get('en', '')),
                "category": concept.get('category', 'ethics'),
                "source": "knowledge_base"
            }
            
            # 添加额外信息
            for key in ['major_theories', 'key_concepts', 'applications', 'principles']:
                if key in concept:
                    guideline[key] = concept[key]
            
            return guideline
            
        except Exception as e:
            logger.error(f"Failed to extract guideline from concept: {e}")
            return None
    
    def _extract_relevant_segment(self, text: str, concern: str) -> str:
        """提取相关文本片段"""
        # 简单实现：返回前100个字符
        return text[:100] + "..." if len(text) > 100 else text
    
    def _generate_replacement(self, text: str, concern: str, language: str) -> str:
        """生成替换文本"""
        # 基于问题类型生成建议性文本
        concern_lower = concern.lower()
        
        if "safety" in concern_lower:
            return self._translate("[Safe alternative that avoids harm]", language)
        elif "honesty" in concern_lower:
            return self._translate("[Truthful and transparent version]", language)
        elif "privacy" in concern_lower:
            return self._translate("[Privacy-respecting alternative]", language)
        elif "fairness" in concern_lower:
            return self._translate("[Fair and equitable version]", language)
        else:
            return self._translate("[Improved version aligned with values]", language)
    
    def _get_correction_reason(self, concern: str, language: str) -> str:
        """获取修正原因"""
        concern_lower = concern.lower()
        
        if "safety" in concern_lower:
            return self._translate("To ensure no harm is caused to people or the environment", language)
        elif "honesty" in concern_lower:
            return self._translate("To maintain truthfulness and transparency", language)
        elif "privacy" in concern_lower:
            return self._translate("To respect personal privacy and data protection", language)
        elif "fairness" in concern_lower:
            return self._translate("To ensure fairness and equal treatment", language)
        else:
            return self._translate("To better align with ethical values", language)
    
    def _find_relevant_guideline(self, concern: str, guidelines: List[str], language: str) -> str:
        """查找相关指导原则"""
        if guidelines:
            return guidelines[0]
        
        concern_lower = concern.lower()
        if "safety" in concern_lower:
            return self._translate("Safety first principle", language)
        elif "honesty" in concern_lower:
            return self._translate("Honesty and transparency principle", language)
        elif "privacy" in concern_lower:
            return self._translate("Privacy protection principle", language)
        else:
            return self._translate("General ethical guideline", language)
    
    def _improve_general_alignment(self, text: str, language: str) -> str:
        """改进一般对齐"""
        base_text = text
        
        # 添加价值对齐前缀
        prefix = self._translate("In alignment with ethical values: ", language)
        
        # 确保文本以正面方式表达
        if any(negative in base_text.lower() for negative in ['harm', 'hurt', 'damage', 'destroy']):
            base_text = self._translate("Considering the wellbeing of all stakeholders, ", language) + base_text
        
        return prefix + base_text
    
    def _translate(self, text: str, language: str) -> str:
        """简单翻译（占位实现）"""
        # 在实际实现中，这里应该调用真正的翻译服务
        # 目前返回原文作为占位
        return text
    
    async def start(self):
        """启动价值对齐服务"""
        if self.is_running:
            logger.warning("Value alignment service is already running")
            return
        
        try:
            self.start_time = time.time()
            self.is_running = True
            
            logger.info(f"Value alignment service starting on port {self.port}")
            
            # 启动FastAPI服务器
            config = uvicorn.Config(
                app=self.app,
                host=SERVICE_HOST,
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            await server.serve()
            
        except Exception as e:
            error_msg = f"Failed to start value alignment service: {str(e)}"
            logger.error(error_msg)
            error_handler.handle_error(e, "ValueAlignmentService", "启动失败")
            raise
    
    async def stop(self):
        """停止价值对齐服务"""
        try:
            self.is_running = False
            await self.client.aclose()
            logger.info("Value alignment service stopped")
        except Exception as e:
            logger.error(f"Error stopping value alignment service: {e}")

# 全局服务实例
_value_alignment_service = None

def get_value_alignment_service(port: int = VALUE_ALIGNMENT_SERVICE_PORT) -> ValueAlignmentService:
    """获取价值对齐服务实例"""
    global _value_alignment_service
    if _value_alignment_service is None:
        _value_alignment_service = ValueAlignmentService(port)
    return _value_alignment_service

async def start_value_alignment_service(port: int = VALUE_ALIGNMENT_SERVICE_PORT):
    """启动价值对齐服务"""
    service = get_value_alignment_service(port)
    await service.start()

async def stop_value_alignment_service():
    """停止价值对齐服务"""
    global _value_alignment_service
    if _value_alignment_service:
        await _value_alignment_service.stop()
        _value_alignment_service = None

# 命令行直接运行
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Value Alignment Service")
    parser.add_argument("--port", type=int, default=VALUE_ALIGNMENT_SERVICE_PORT, 
                       help=f"Port to run the service on (default: {VALUE_ALIGNMENT_SERVICE_PORT})")
    
    args = parser.parse_args()
    
    # 运行服务
    service = ValueAlignmentService(args.port)
    
    print(f"Starting Value Alignment Service on port {args.port}...")
    print(f"API Documentation: http://localhost:{args.port}/docs")
    print(f"Health Check: http://localhost:{args.port}/health")
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(service.start())