"""
错误处理API端点
Error Handling API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .production_error_handler import (
    ProductionErrorHandler, get_error_handler, ErrorSeverity, ErrorCategory, ErrorContext
)
from .production_monitoring import ProductionLogger

router = APIRouter(prefix="/api/errors", tags=["error-handling"])

class ErrorSummaryResponse(BaseModel):
    """错误摘要响应模型"""
    timestamp: str
    total_errors: int
    recovered_errors: int
    unrecovered_errors: int
    errors_by_severity: Dict[str, int]
    errors_by_category: Dict[str, int]
    recent_errors: List[Dict[str, Any]]

class ErrorDetailsResponse(BaseModel):
    """错误详情响应模型"""
    error_id: str
    severity: str
    category: str
    message: str
    exception_type: str
    exception_message: str
    stack_trace: str
    component: str
    operation: str
    user_id: Optional[str]
    request_id: Optional[str]
    timestamp: str
    retry_count: int
    handled: bool
    recovery_action: Optional[str]

class ClearErrorsRequest(BaseModel):
    """清空错误请求模型"""
    confirm: bool

@router.get("/summary", response_model=ErrorSummaryResponse)
async def get_error_summary(
    error_handler: ProductionErrorHandler = Depends(get_error_handler)
):
    """获取错误摘要"""
    try:
        summary = error_handler.get_error_summary()
        return ErrorSummaryResponse(**summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取错误摘要失败: {str(e)}")

@router.get("/details/{error_id}", response_model=ErrorDetailsResponse)
async def get_error_details(
    error_id: str,
    error_handler: ProductionErrorHandler = Depends(get_error_handler)
):
    """获取特定错误的详细信息"""
    try:
        error_details = error_handler.error_history.get(error_id)
        if not error_details:
            raise HTTPException(status_code=404, detail="错误ID不存在")
        
        # 转换为响应模型
        return ErrorDetailsResponse(
            error_id=error_details.error_id,
            severity=error_details.severity.value,
            category=error_details.category.value,
            message=error_details.message,
            exception_type=error_details.exception_type,
            exception_message=error_details.exception_message,
            stack_trace=error_details.stack_trace,
            component=error_details.context.component,
            operation=error_details.context.operation,
            user_id=error_details.context.user_id,
            request_id=error_details.context.request_id,
            timestamp=error_details.context.timestamp.isoformat(),
            retry_count=error_details.retry_count,
            handled=error_details.handled,
            recovery_action=error_details.recovery_action
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取错误详情失败: {str(e)}")

@router.get("/list", response_model=List[Dict[str, Any]])
async def list_errors(
    limit: int = 50,
    severity: Optional[str] = None,
    category: Optional[str] = None,
    handled: Optional[bool] = None,
    error_handler: ProductionErrorHandler = Depends(get_error_handler)
):
    """列出错误（支持过滤）"""
    try:
        errors = []
        
        for error_details in error_handler.error_history.values():
            # 应用过滤器
            if severity and error_details.severity.value != severity:
                continue
            if category and error_details.category.value != category:
                continue
            if handled is not None and error_details.handled != handled:
                continue
            
            errors.append({
                "error_id": error_details.error_id,
                "severity": error_details.severity.value,
                "category": error_details.category.value,
                "message": error_details.message,
                "exception_type": error_details.exception_type,
                "timestamp": error_details.context.timestamp.isoformat(),
                "component": error_details.context.component,
                "operation": error_details.context.operation,
                "handled": error_details.handled
            })
            
            if len(errors) >= limit:
                break
        
        # 按时间倒序排序
        errors.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"列出错误失败: {str(e)}")

@router.post("/clear")
async def clear_errors(
    request: ClearErrorsRequest,
    error_handler: ProductionErrorHandler = Depends(get_error_handler)
):
    """清空错误历史"""
    try:
        if not request.confirm:
            raise HTTPException(status_code=400, detail="需要确认操作")
        
        error_handler.clear_error_history()
        return {"message": "错误历史已清空"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空错误失败: {str(e)}")

@router.get("/stats")
async def get_error_stats(
    error_handler: ProductionErrorHandler = Depends(get_error_handler)
):
    """获取错误统计信息"""
    try:
        stats = error_handler.error_stats
        
        # 计算错误率（基于时间窗口）
        total_errors = stats["total_errors"]
        recovered_rate = (
            stats["recovered_errors"] / total_errors * 100 
            if total_errors > 0 else 100
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_errors": total_errors,
            "recovered_errors": stats["recovered_errors"],
            "unrecovered_errors": stats["unrecovered_errors"],
            "recovery_rate": round(recovered_rate, 2),
            "errors_by_severity": stats["errors_by_severity"],
            "errors_by_category": stats["errors_by_category"],
            "error_history_size": len(error_handler.error_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取错误统计失败: {str(e)}")

@router.post("/test/{error_type}")
async def test_error_handling(
    error_type: str,
    error_handler: ProductionErrorHandler = Depends(get_error_handler)
):
    """测试错误处理功能"""
    try:
        from .production_error_handler import ErrorContext
        
        # 创建测试错误上下文
        context = ErrorContext(
            timestamp=datetime.now(),
            component="test_api",
            operation="test_error_handling",
            user_id="test_user",
            request_id="test_request_123"
        )
        
        # 根据错误类型创建不同的异常
        if error_type == "connection":
            exception = ConnectionError("测试连接错误")
            severity = ErrorSeverity.HIGH
            category = ErrorCategory.NETWORK
            retryable = True
        elif error_type == "database":
            exception = Exception("测试数据库错误")
            severity = ErrorSeverity.HIGH
            category = ErrorCategory.DATABASE
            retryable = True
        elif error_type == "validation":
            exception = ValueError("测试验证错误")
            severity = ErrorSeverity.MEDIUM
            category = ErrorCategory.VALIDATION
            retryable = False
        elif error_type == "system":
            exception = RuntimeError("测试系统错误")
            severity = ErrorSeverity.CRITICAL
            category = ErrorCategory.SYSTEM
            retryable = False
        else:
            raise HTTPException(status_code=400, detail="不支持的错误类型")
        
        # 处理错误
        error_details = error_handler.handle_error(
            exception, context, severity, category, retryable
        )
        
        return {
            "message": "错误处理测试完成",
            "error_id": error_details.error_id,
            "severity": error_details.severity.value,
            "category": error_details.category.value,
            "handled": error_details.handled,
            "recovery_action": error_details.recovery_action
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"错误处理测试失败: {str(e)}")

# 全局错误处理中间件
async def global_error_handler_middleware(request, call_next):
    """全局错误处理中间件"""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # 获取错误处理器
        try:
            error_handler = get_error_handler()
            
            # 创建错误上下文
            context = ErrorContext(
                timestamp=datetime.now(),
                component="fastapi",
                operation=request.method + " " + request.url.path,
                user_id=request.headers.get("user_id"),
                request_id=request.headers.get("x-request-id"),
                session_id=request.cookies.get("session_id")
            )
            
            # 确定错误严重程度和类别
            severity = ErrorSeverity.HIGH
            category = ErrorCategory.API
            
            # 处理错误
            error_details = error_handler.handle_error(e, context, severity, category)
            
            # 根据错误详情返回适当的HTTP响应
            if error_details.handled:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal Server Error",
                        "message": "服务暂时不可用，请稍后重试",
                        "error_id": error_details.error_id
                    }
                )
            else:
                # 重新抛出未处理的错误
                raise
                
        except Exception as handler_error:
            # 错误处理器本身出错，返回通用错误
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "系统错误，请联系管理员"
                }
            )

# 在FastAPI应用中集成错误处理
def setup_error_handling(app, error_handler):
    """设置错误处理"""
    # 使用正确的logger对象
    import logging
    logger = logging.getLogger("error_handler")
    
    # 添加错误处理API路由
    app.include_router(router)
    
    # 添加全局错误处理中间件
    @app.middleware("http")
    async def error_handling_middleware(request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # 使用错误处理器处理异常
            context = ErrorContext(
                timestamp=datetime.now(),
                component="middleware",
                operation=request.method + " " + request.url.path
            )
            
            error_details = error_handler.handle_error(
                e, context, ErrorSeverity.MEDIUM, ErrorCategory.MIDDLEWARE
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "系统错误，请联系管理员",
                    "error_id": error_details.get("error_id", "unknown") if isinstance(error_details, dict) else (error_details.error_id if error_details else "unknown")
                }
            )
    
    # 添加自定义异常处理器
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Error",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        # 使用错误处理器处理通用异常
        context = ErrorContext(
            timestamp=datetime.now(),
            component="fastapi",
            operation=request.method + " " + request.url.path
        )
        
        error_details = error_handler.handle_error(
            exc, context, ErrorSeverity.HIGH, ErrorCategory.API
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "系统错误，请联系管理员",
                "error_id": error_details.get("error_id", "unknown") if isinstance(error_details, dict) else (error_details.error_id if error_details else "unknown")
            }
        )
    
    logger.info("错误处理系统已设置完成")
    return error_handler

# 导入必要的依赖
from fastapi.responses import JSONResponse
from .production_error_handler import initialize_error_handler