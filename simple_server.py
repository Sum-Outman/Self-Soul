import sys
import os
import asyncio
from fastapi import FastAPI
import uvicorn

# 设置环境变量以禁用不必要的初始化
os.environ["SKIP_FULL_INITIALIZATION"] = "true"

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_minimal_app():
    """创建一个最小化的FastAPI应用，只包含健康检查端点"""
    app = FastAPI(title="AGI Soul Minimal Server", version="1.0")
    
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "1.0", "message": "Minimal AGI server running"}
    
    @app.get("/")
    async def root():
        return {"message": "Welcome to AGI Soul", "endpoints": ["/health"]}
    
    return app

if __name__ == "__main__":
    try:
        print("=== AGI Soul Minimal Server ===")
        print(f"Python version: {sys.version}")
        print(f"Current directory: {os.getcwd()}")
        
        # 创建最小化应用
        app = create_minimal_app()
        print(f"\n✓ Created minimal FastAPI app")
        print(f"✓ Number of routes: {len(app.routes)}")
        
        # 打印路由信息
        print("\nAvailable routes:")
        for route in app.routes:
            if hasattr(route, 'path'):
                print(f"   - {route.path} ({route.name})")
        
        # 启动服务器
        print("\nStarting server on http://127.0.0.1:8000...")
        print("Press Ctrl+C to stop")
        
        # 使用简单的配置启动服务器
        uvicorn.run(
            app, 
            host="127.0.0.1", 
            port=8000,
            reload=False,
            workers=1,
            loop="asyncio"
        )
        
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nMinimal server process completed.")