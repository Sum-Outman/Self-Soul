#!/usr/bin/env python3
"""
渐进式FastAPI服务器 - 逐步集成AGI系统功能
"""

import uvicorn
import asyncio
import logging
import threading
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from datetime import datetime
import sys
import os

# 加载环境变量
try:
    from dotenv import load_dotenv
    # 加载.env文件（如果存在）
    load_dotenv()
    print("环境变量已从.env文件加载")
except ImportError:
    print("警告：未找到python-dotenv，环境变量将仅从系统环境读取")

# 添加核心模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入端口
from core.model_ports_config import MAIN_API_PORT

# 导入核心模型
try:
    from core.self_model import GoalModel
except ImportError as e:
    print(f"Warning: Could not import GoalModel: {e}")
    GoalModel = None

# 导入知识服务
try:
    from core.knowledge_service import get_knowledge_service

    knowledge_service = get_knowledge_service()
except ImportError as e:
    print(f"Warning: Could not import knowledge service: {e}")
    knowledge_service = None

# 导入系统监控
try:
    from core.system_monitor import get_realtime_metrics, monitor
    has_monitoring = True
except ImportError as e:
    print(f"Warning: Could not import system monitoring: {e}")
    has_monitoring = False

# 导入机器人API增强模块
try:
    from core.robot_api_enhanced import router as robot_enhanced_router, initialize_enhanced_robot_api
    robot_enhanced_available = True
    print("机器人API增强模块导入成功")
except ImportError as e:
    print(f"Warning: Could not import robot enhanced API: {e}")
    robot_enhanced_available = False

# 创建FastAPI应用
app = FastAPI(title="AGI System - Progressive Server", version="1.0.0")

# 添加机器人API增强路由
if robot_enhanced_available:
    app.include_router(robot_enhanced_router)
    print("机器人API增强路由已添加")

# 线程安全的共享状态管理

class SharedState:
    """线程安全的共享状态管理类"""

    def __init__(self):
        self._lock = threading.RLock()  # 可重入锁，支持嵌套调用
        self._model_registry = None
        self._language_model = None
        self._goal_model = None

    @property
    def model_registry(self):
        """线程安全获取模型注册表"""
        with self._lock:
            return self._model_registry

    @model_registry.setter
    def model_registry(self, value):
        """线程安全设置模型注册表"""
        with self._lock:
            self._model_registry = value

    @property
    def language_model(self):
        """线程安全获取语言模型"""
        with self._lock:
            return self._language_model

    @language_model.setter
    def language_model(self, value):
        """线程安全设置语言模型"""
        with self._lock:
            self._language_model = value

    @property
    def goal_model(self):
        """线程安全获取目标模型"""
        with self._lock:
            return self._goal_model

    @goal_model.setter
    def goal_model(self, value):
        """线程安全设置目标模型"""
        with self._lock:
            self._goal_model = value

    def get_components_status(self) -> Dict[str, bool]:
        """获取组件加载状态"""
        with self._lock:
            return {
                "model_registry": self._model_registry is not None,
                "language_model": self._language_model is not None,
                "goal_model": self._goal_model is not None,
            }


# 共享状态实例
shared_state = SharedState()
logger = logging.getLogger("progressive_server")

# 阶段1: 基本功能
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Progressive server is running",
        "stage": "basic",
    }


@app.get("/api/system/status")
async def system_status():
    return {
        "status": "ok",
        "system": "progressive",
        "version": "1.0.0",
        "stage": "basic",
        "components_loaded": shared_state.get_components_status(),
    }


# 阶段2: 聊天功能（集成语言模型）
@app.post("/api/chat")
async def chat_endpoint(request: Dict[str, Any]):
    try:
        # 支持text和message两种参数名
        text = request.get("text", request.get("message", ""))

        if not text or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Invalid or empty text input")

        # 使用真正的语言模型处理聊天请求
        model_result = await process_chat_with_model(
            text, {"request_id": f"req_{datetime.now().timestamp()}"}
        )

        # 如果语言模型不可用或处理失败，回退到基本响应
        if not model_result.get("status") == "success" or not model_result.get(
            "model_loaded", False
        ):
            logger.warning(f"语言模型处理失败或未加载，使用基本响应: {model_result}")
            # 生成基本响应作为后备
            message_lower = text.lower()
            if "hello" in message_lower or "hi" in message_lower:
                response_text = "Hello! I'm an AI assistant. How can I help you today?"
            elif "how are you" in message_lower:
                response_text = "I'm doing well, thank you! I'm here to assist you with any questions or tasks you have."
            elif "what can you do" in message_lower:
                response_text = "I can help you with various tasks like answering questions, providing information, and assisting with problems. What would you like help with?"
            elif "thank you" in message_lower or "thanks" in message_lower:
                response_text = "You're welcome! It was my pleasure to assist you."
            elif "bye" in message_lower or "goodbye" in message_lower:
                response_text = "Goodbye! Have a great day."
            else:
                response_text = f"I've received your message: '{text}'. I'm an AI assistant ready to help you."

            model_mode = "basic_fallback"
        else:
            # 使用语言模型的响应
            response_text = model_result.get("response", "")
            model_mode = "advanced_model"

        # 返回与其他API端点一致的格式
        return {
            "status": "success",
            "data": {
                "response": response_text,
                "response_type": "text",
                "conversation_history": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": response_text},
                ],
                "session_id": f"session_{datetime.now().timestamp()}",
                "model_mode": model_mode,
                "model_loaded": model_result.get("model_loaded", False),
            },
            "mode": model_mode,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


async def process_chat_with_model(
    text: str, context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """使用语言模型处理聊天"""
    try:
        if not shared_state.language_model:
            return {
                "status": "error",
                "response": "Language model is not available",
                "stage": "advanced",
                "model_loaded": False,
            }

        # 使用真正的语言模型处理请求
        input_data = {"text": text, "context": context or {}}

        result = shared_state.language_model.process_input(input_data)

        if result.get("success", False):
            return {
                "status": "success",
                "response": result["response"],
                "stage": "advanced",
                "model_loaded": True,
                "emotion_state": result.get("emotion_state", {}),
            }
        else:
            return {
                "status": "error",
                "response": result.get("error", "Model processing failed"),
                "stage": "advanced",
                "model_loaded": True,
            }
    except Exception as e:
        return {
            "status": "error",
            "response": f"Model processing error: {str(e)}",
            "stage": "advanced",
            "model_loaded": shared_state.language_model is not None,
        }


# AGI规划和推理API端点
@app.post("/api/agi/plan-with-reasoning")
async def plan_with_reasoning_endpoint(request: Dict[str, Any]):
    """使用AGI级规划推理引擎生成计划"""
    try:
        goal = request.get("goal", "")
        context = request.get("context", {})
        constraints = request.get("constraints", {})
        available_resources = request.get("available_resources", [])

        if not goal:
            raise HTTPException(status_code=400, detail="Goal is required")

        # 延迟导入以避免启动时依赖问题
        try:
            from core.advanced_reasoning import EnhancedAdvancedReasoningEngine

            engine = EnhancedAdvancedReasoningEngine()

            # 使用集成规划推理
            result = engine.plan_with_reasoning(
                goal, context, constraints, available_resources
            )

            return {
                "status": "success"
                if result.get("success", False)
                else "partial_success",
                "data": result,
                "planning_mode": "integrated_agi_reasoning",
                "timestamp": datetime.now().isoformat(),
            }
        except ImportError as e:
            logger.warning(f"EnhancedAdvancedReasoningEngine not available: {e}")
            return {
                "status": "error",
                "error": "AGI规划推理引擎未加载",
                "planning_mode": "basic_fallback",
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Planning with reasoning error: {str(e)}"
        )


@app.post("/api/agi/analyze-causality")
async def analyze_causality_endpoint(request: Dict[str, Any]):
    """分析计划的因果结构"""
    try:
        plan = request.get("plan", {})
        context = request.get("context", {})

        if not plan:
            raise HTTPException(status_code=400, detail="Plan is required")

        # 延迟导入以避免启动时依赖问题
        try:
            from core.advanced_reasoning import EnhancedAdvancedReasoningEngine

            engine = EnhancedAdvancedReasoningEngine()

            # 分析因果结构
            result = engine.analyze_causality(plan, context)

            return {
                "status": "success"
                if result.get("success", False)
                else "partial_success",
                "data": result,
                "analysis_mode": "causal_reasoning",
                "timestamp": datetime.now().isoformat(),
            }
        except ImportError as e:
            logger.warning(f"EnhancedAdvancedReasoningEngine not available: {e}")
            return {
                "status": "error",
                "error": "因果推理引擎未加载",
                "analysis_mode": "basic_fallback",
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Causal analysis error: {str(e)}")


@app.post("/api/agi/temporal-planning")
async def temporal_planning_endpoint(request: Dict[str, Any]):
    """时间约束下的规划"""
    try:
        goal = request.get("goal", "")
        temporal_constraints = request.get("temporal_constraints", {})
        context = request.get("context", {})

        if not goal:
            raise HTTPException(status_code=400, detail="Goal is required")
        if not temporal_constraints:
            raise HTTPException(
                status_code=400, detail="Temporal constraints are required"
            )

        # 延迟导入以避免启动时依赖问题
        try:
            from core.temporal_reasoning_planner import (
                create_temporal_reasoning_planner,
            )

            planner = create_temporal_reasoning_planner()

            # 时间约束规划
            result = planner.plan_with_temporal_constraints(goal, temporal_constraints)

            return {
                "status": "success"
                if result.get("success", False)
                else "partial_success",
                "data": result,
                "planning_mode": "temporal_reasoning",
                "timestamp": datetime.now().isoformat(),
            }
        except ImportError as e:
            logger.warning(f"TemporalReasoningPlanner not available: {e}")
            return {
                "status": "error",
                "error": "时间推理规划器未加载",
                "planning_mode": "basic_fallback",
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Temporal planning error: {str(e)}"
        )


@app.post("/api/agi/cross-domain-planning")
async def cross_domain_planning_endpoint(request: Dict[str, Any]):
    """跨领域规划"""
    try:
        goal = request.get("goal", "")
        target_domain = request.get("target_domain", "")
        context = request.get("context", {})
        available_domains = request.get("available_domains", [])
        constraints = request.get("constraints", {})

        if not goal:
            raise HTTPException(status_code=400, detail="Goal is required")
        if not target_domain:
            raise HTTPException(status_code=400, detail="Target domain is required")

        # 延迟导入以避免启动时依赖问题
        try:
            from core.cross_domain_planner import create_cross_domain_planner

            planner = create_cross_domain_planner()

            # 跨领域规划
            result = planner.plan_across_domains(
                goal=goal,
                target_domain=target_domain,
                context=context,
                available_domains=available_domains,
                constraints=constraints,
            )

            return {
                "status": "success"
                if result.get("success", False)
                else "partial_success",
                "data": result,
                "planning_mode": "cross_domain",
                "timestamp": datetime.now().isoformat(),
            }
        except ImportError as e:
            logger.warning(f"CrossDomainPlanner not available: {e}")
            return {
                "status": "error",
                "error": "跨领域规划器未加载",
                "planning_mode": "basic_fallback",
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Cross-domain planning error: {str(e)}"
        )


@app.post("/api/agi/self-reflection")
async def self_reflection_endpoint(request: Dict[str, Any]):
    """自我反思和优化"""
    try:
        reflection_type = request.get("reflection_type", "performance")
        data = request.get("data", {})
        context = request.get("context", {})

        if not data:
            raise HTTPException(
                status_code=400, detail="Data is required for reflection"
            )

        # 延迟导入以避免启动时依赖问题
        try:
            from core.self_reflection_optimizer import create_self_reflection_optimizer

            optimizer = create_self_reflection_optimizer()

            # 根据反思类型执行不同的反思
            if reflection_type == "performance":
                result = optimizer.reflect_on_performance(data, context)
            elif reflection_type == "errors":
                result = optimizer.reflect_on_errors(data, context)
            elif reflection_type == "strategy":
                performance_outcomes = request.get("performance_outcomes", {})
                result = optimizer.reflect_on_strategy(
                    data, performance_outcomes, context
                )
            else:
                return {
                    "status": "error",
                    "error": f"Unknown reflection type: {reflection_type}",
                    "timestamp": datetime.now().isoformat(),
                }

            return {
                "status": "success"
                if result.get("success", False)
                else "partial_success",
                "data": result,
                "reflection_mode": reflection_type,
                "timestamp": datetime.now().isoformat(),
            }
        except ImportError as e:
            logger.warning(f"SelfReflectionOptimizer not available: {e}")
            return {
                "status": "error",
                "error": "自我反思优化器未加载",
                "reflection_mode": "basic_fallback",
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Self-reflection error: {str(e)}")


# 启动事件处理
@app.on_event("startup")
async def startup_event():
    """渐进式启动 - 先启动基本功能，再逐步加载组件"""
    print("Starting progressive AGI server...")

    # 阶段1: 基本服务器功能
    print("Stage 1: Basic server functionality - COMPLETE")

    # 阶段2: 在后台逐步加载组件
    loop = asyncio.get_event_loop()
    loop.create_task(load_components_gradually())


async def load_components_gradually():
    """逐步加载AGI系统组件"""
    print("Stage 2: Starting gradual component loading...")

    try:
        # 第一步: 加载模型注册表
        print("  Loading model registry...")
        await load_model_registry()

        # 第二步: 加载语言模型
        print("  Loading language model...")
        await load_language_model()

        # 第三步: 加载其他组件
        print("  Loading other components...")
        await load_other_components()

        print("Stage 2: Component loading - COMPLETE")

    except Exception as e:
        print(f"Component loading failed: {e}")


async def load_model_registry():
    """加载模型注册表"""
    try:
        from core.model_registry import ModelRegistry

        shared_state.model_registry = ModelRegistry()
        print("    Model registry loaded successfully")
    except Exception as e:
        print(f"    Failed to load model registry: {e}")


async def load_language_model():
    """加载语言模型"""
    try:
        # 加载真正的语言模型
        from core.models.language.unified_language_model import UnifiedLanguageModel

        shared_state.language_model = UnifiedLanguageModel()
        print("    Language model loaded successfully")
    except Exception as e:
        print(f"    Failed to load language model: {e}")


async def load_other_components():
    """加载其他组件"""
    try:
        # 尝试加载AGI规划和推理组件
        print("    Loading AGI planning and reasoning components...")

        try:
            from core.integrated_planning_reasoning_engine import (
                create_integrated_planning_reasoning_engine,
            )

            # 注意：这里创建引擎但不存储，端点会自己创建
            print("      Integrated planning reasoning engine available")
        except ImportError as e:
            print(f"      Integrated planning reasoning engine not available: {e}")

        try:
            from core.causal_reasoning_enhancer import create_causal_reasoning_enhancer

            print("      Causal reasoning enhancer available")
        except ImportError as e:
            print(f"      Causal reasoning enhancer not available: {e}")

        try:
            from core.temporal_reasoning_planner import (
                create_temporal_reasoning_planner,
            )

            print("      Temporal reasoning planner available")
        except ImportError as e:
            print(f"      Temporal reasoning planner not available: {e}")

        try:
            from core.cross_domain_planner import create_cross_domain_planner

            print("      Cross domain planner available")
        except ImportError as e:
            print(f"      Cross domain planner not available: {e}")

        try:
            from core.self_reflection_optimizer import create_self_reflection_optimizer

            print("      Self reflection optimizer available")
        except ImportError as e:
            print(f"      Self reflection optimizer not available: {e}")

        print("    Other components loaded")
    except Exception as e:
        print(f"    Failed to load other components: {e}")


# Goal management API endpoints
@app.get("/api/goals")
async def get_goals():
    """Get current goals and their status"""
    try:
        if GoalModel is None:
            raise HTTPException(
                status_code=501, detail="GoalModel module not available"
            )

        # Initialize goal model if not already done
        if shared_state.goal_model is None:
            shared_state.goal_model = GoalModel(from_scratch=True)

        goal_model = shared_state.goal_model
        report = goal_model.get_goal_report()

        return {
            "status": "success",
            "data": report,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get goals: {str(e)}")


@app.get("/api/goals/critical")
async def get_critical_goals():
    """Get critical goals that need attention"""
    try:
        if GoalModel is None:
            raise HTTPException(
                status_code=501, detail="GoalModel module not available"
            )

        if shared_state.goal_model is None:
            shared_state.goal_model = GoalModel(from_scratch=True)

        goal_model = shared_state.goal_model
        critical_goals = goal_model.identify_critical_goals()

        return {
            "status": "success",
            "data": {"critical_goals": critical_goals, "count": len(critical_goals)},
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get critical goals: {str(e)}"
        )


# Engineering Knowledge API endpoints
@app.get("/api/knowledge/domains")
async def get_knowledge_domains():
    """Get list of available engineering knowledge domains"""
    try:
        if knowledge_service is None:
            raise HTTPException(
                status_code=501, detail="Knowledge service not available"
            )

        domains = knowledge_service.get_domains()
        domain_info = []

        for domain in domains:
            info = knowledge_service.get_domain_info(domain)
            if info:
                domain_info.append(info)

        return {
            "status": "success",
            "data": {"domains": domain_info, "count": len(domain_info)},
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get knowledge domains: {str(e)}"
        )


@app.get("/api/knowledge/search")
async def search_knowledge(query: str = "", domain: str = None):
    """Search for concepts in engineering knowledge bases"""
    try:
        if knowledge_service is None:
            raise HTTPException(
                status_code=501, detail="Knowledge service not available"
            )

        if not query and not domain:
            return {
                "status": "error",
                "message": "Either query or domain parameter is required",
            }

        results = knowledge_service.search_concepts(query, domain)

        return {
            "status": "success",
            "data": {
                "results": results,
                "count": len(results),
                "query": query,
                "domain": domain,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to search knowledge: {str(e)}"
        )


@app.get("/api/knowledge/concept/{domain}/{concept_id}")
async def get_concept_detail(domain: str, concept_id: str):
    """Get detailed information about a specific concept"""
    try:
        if knowledge_service is None:
            raise HTTPException(
                status_code=501, detail="Knowledge service not available"
            )

        concept = knowledge_service.get_concept(domain, concept_id)

        if not concept:
            raise HTTPException(
                status_code=404,
                detail=f"Concept '{concept_id}' not found in domain '{domain}'",
            )

        return {
            "status": "success",
            "data": concept,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get concept: {str(e)}")


@app.get("/api/knowledge/statistics")
async def get_knowledge_statistics():
    """Get statistics about loaded engineering knowledge"""
    try:
        if knowledge_service is None:
            raise HTTPException(
                status_code=501, detail="Knowledge service not available"
            )

        stats = knowledge_service.get_statistics()

        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get knowledge statistics: {str(e)}"
        )


# Meta-cognition API endpoints
@app.post("/api/meta-cognition/analyze")
async def analyze_meta_cognition(request: Dict[str, Any]):
    """Analyze cognitive processes and provide meta-cognitive insights"""
    try:
        # Try to import meta-cognition module
        try:
            from core.enhanced_meta_cognition import EnhancedMetaCognition

            meta_cognition = EnhancedMetaCognition()
        except ImportError:
            return {
                "status": "info",
                "message": "Meta-cognition module not fully implemented yet",
                "data": {
                    "analysis": "Placeholder analysis - module under development",
                    "insights": [
                        "Cognitive patterns detected",
                        "Learning optimization opportunities",
                    ],
                    "recommendations": [
                        "Implement full meta-cognition logic",
                        "Add learning rate adaptation",
                    ],
                },
                "timestamp": datetime.now().isoformat(),
            }

        # Process request if module is available
        cognitive_data = request.get("cognitive_data", {})
        analysis_result = meta_cognition.analyze_cognitive_processes(cognitive_data)

        return {
            "status": "success",
            "data": analysis_result,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Meta-cognition analysis failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/api/meta-cognition/status")
async def get_meta_cognition_status():
    """Get meta-cognition system status"""
    return {
        "status": "success",
        "data": {
            "module": "enhanced_meta_cognition",
            "implementation_status": "partial",
            "available_methods": ["cognitive_analysis", "learning_optimization"],
            "capabilities": [
                "Pattern recognition",
                "Learning rate adaptation",
                "Cognitive load analysis",
            ],
        },
        "timestamp": datetime.now().isoformat(),
    }


# Explainable AI API endpoints
@app.post("/api/explainable-ai/explain")
async def explain_ai_decision(request: Dict[str, Any]):
    """Explain AI decisions and provide interpretable insights"""
    try:
        # Try to import explainable AI module
        try:
            from core.explainable_ai import DecisionTracer

            decision_tracer = DecisionTracer()
        except ImportError:
            return {
                "status": "info",
                "message": "Explainable AI module not fully implemented yet",
                "data": {
                    "explanation": "Placeholder explanation - module under development",
                    "decision_factors": [
                        "Input patterns",
                        "Historical context",
                        "Confidence thresholds",
                    ],
                    "interpretability_score": 0.7,
                },
                "timestamp": datetime.now().isoformat(),
            }

        # Process request if module is available
        decision_data = request.get("decision_data", {})
        explanation = decision_tracer.explain_decision(decision_data)

        return {
            "status": "success",
            "data": explanation,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"AI decision explanation failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/api/explainable-ai/capabilities")
async def get_explainable_ai_capabilities():
    """Get explainable AI system capabilities"""
    return {
        "status": "success",
        "data": {
            "module": "explainable_ai",
            "implementation_status": "partial",
            "available_methods": ["decision_tracing", "explanation_generation"],
            "capabilities": [
                "Decision transparency",
                "Confidence scoring",
                "Interpretable insights",
            ],
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/monitoring/data")
async def get_monitoring_data():
    """Get real-time monitoring data for frontend dashboard"""
    if not has_monitoring:
        return {
            "status": "success",
            "data": {
                "systemStatus": "normal",
                "systemUptime": "00:00:00",
                "activeModelsCount": 0,
                "totalModelsCount": 0,
                "cpuUsage": 0,
                "memoryUsage": 0,
                "modelMetrics": [],
                "dataStream": []
            },
            "message": "Monitoring module not available - returning default data",
            "timestamp": datetime.now().isoformat(),
        }
    
    try:
        # Get real-time metrics from monitoring system
        realtime_data = monitor.get_realtime_monitoring()
        
        # Transform data to match frontend expectations
        system_data = realtime_data.get("system", {})
        models_data = realtime_data.get("models", {})
        
        # Format model metrics for frontend
        model_metrics = []
        if "model_metrics" in models_data:
            for model_name, metrics in models_data["model_metrics"].items():
                model_metrics.append({
                    "name": "accuracy",
                    "value": f"{metrics.get('accuracy', 0):.2f}%",
                    "trend": "stable",
                    "change": "0%"
                })
                model_metrics.append({
                    "name": "latency",
                    "value": f"{metrics.get('latency_ms', 0):.0f}ms",
                    "trend": "stable",
                    "change": "0ms"
                })
                model_metrics.append({
                    "name": "throughput",
                    "value": f"{metrics.get('throughput', 0):.1f}/s",
                    "trend": "stable",
                    "change": "0/s"
                })
        
        # Format data stream
        data_stream = []
        if "logs" in realtime_data:
            recent_logs = realtime_data["logs"].get("recent_logs", [])
            for log in recent_logs[-10:]:  # Last 10 logs
                data_stream.append({
                    "timestamp": log.get("timestamp", ""),
                    "model": log.get("source", "system"),
                    "type": log.get("level", "info").lower(),
                    "details": log.get("message", "")
                })
        
        # Return formatted data for frontend
        return {
            "status": "success",
            "data": {
                "systemStatus": "normal",  # Would come from health checks
                "systemUptime": f"{system_data.get('uptime', 0):.0f}s",
                "activeModelsCount": models_data.get("active_models", 0),
                "totalModelsCount": models_data.get("total_models", 0),
                "cpuUsage": system_data.get("cpu_usage", 0),
                "memoryUsage": system_data.get("memory_usage", 0),
                "modelMetrics": model_metrics,
                "dataStream": data_stream
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to fetch monitoring data: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    print("Progressive AGI System Server")
    print("=" * 50)
    print("Available endpoints:")
    print("  - GET /health")
    print("  - POST /api/chat")
    print("  - GET /api/system/status")
    print("  - POST /api/agi/plan-with-reasoning (AGI规划推理)")
    print("  - POST /api/agi/analyze-causality (因果分析)")
    print("  - POST /api/agi/temporal-planning (时间规划)")
    print("  - POST /api/agi/cross-domain-planning (跨领域规划)")
    print("  - POST /api/agi/self-reflection (自我反思)")
    print("  - GET /api/goals (目标管理)")
    print("  - POST /api/goals/update (更新目标进度)")
    print("  - GET /api/knowledge/domains (工程知识领域)")
    print("  - GET /api/knowledge/concepts (搜索知识概念)")
    print("  - GET /api/knowledge/statistics (知识统计)")
    print("  - POST /api/meta-cognition/analyze (元认知分析)")
    print("  - GET /api/meta-cognition/status (元认知状态)")
    print("  - POST /api/explainable-ai/explain (可解释AI决策)")
    print("  - GET /api/explainable-ai/capabilities (可解释AI能力)")
    print("  - GET /api/monitoring/data (实时监控数据)")
    if robot_enhanced_available:
        print("  - GET /api/robot/enhanced/status (增强机器人状态)")
        print("  - POST /api/robot/enhanced/motion/command (机器人运动命令)")
        print("  - GET /api/robot/enhanced/fusion/status (传感器融合状态)")
        print("  - POST /api/robot/enhanced/fusion/start (启动传感器融合)")
        print("  - POST /api/robot/enhanced/fusion/stop (停止传感器融合)")
        print("  - POST /api/robot/enhanced/fusion/process (传感器数据处理)")
        print("  - GET /api/robot/enhanced/motion/capabilities (运动控制能力)")
        print("  - POST /api/robot/enhanced/emergency/stop (紧急停止)")
        print("  - GET /api/robot/enhanced/multimodal/test (多模态集成测试)")
        print("  - GET /api/robot/enhanced/hardware/simulated (模拟硬件信息)")
        print("  - GET /api/robot/enhanced/test/echo (测试连通性)")
        print("  - GET /api/robot/enhanced/test/integration (增强API集成测试)")
    print("=" * 50)
    print("Server will start with basic functionality")
    print("Components will load gradually in the background")
    print("=" * 50)

    # 启动服务器 - 使用环境变量或默认127.0.0.1
    server_host = os.environ.get("SERVER_HOST", "127.0.0.1")

    uvicorn.run(
        app, host=server_host, port=MAIN_API_PORT, log_level="info", access_log=True
    )
