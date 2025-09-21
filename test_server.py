from fastapi import FastAPI
import uvicorn

# 创建一个非常简单的FastAPI应用
test_app = FastAPI()

@test_app.get("/")
def root():
    return {"message": "测试服务器运行正常"}

@test_app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": "now"}

# 直接运行，不依赖__name__ == "__main__"
print("启动测试服务器，监听 http://127.0.0.1:8001")
uvicorn.run(test_app, host="127.0.0.1", port=8001)