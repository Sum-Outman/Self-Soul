-- Self-Soul-B数据库初始化脚本
-- 创建必要的扩展和表结构

-- 启用常用扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- 创建数据库用户（如果需要）
-- CREATE USER self_soul_user WITH PASSWORD 'secure_password';
-- GRANT ALL PRIVILEGES ON DATABASE self_soul TO self_soul_user;

-- 设置搜索路径
SET search_path TO public;

-- 注意：实际表结构将由应用的ORM/迁移工具创建
-- 这里只创建一些基础表（如果需要）

-- 示例：创建系统配置表
CREATE TABLE IF NOT EXISTS system_config (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value TEXT,
    config_type VARCHAR(50) DEFAULT 'string',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config(config_key);

-- 插入默认配置
INSERT INTO system_config (config_key, config_value, config_type, description)
VALUES 
    ('system_version', '1.0.0', 'string', '系统版本号'),
    ('maintenance_mode', 'false', 'boolean', '维护模式开关'),
    ('max_concurrent_tasks', '10', 'integer', '最大并发任务数'),
    ('default_language', 'en', 'string', '默认语言')
ON CONFLICT (config_key) DO NOTHING;

-- 创建审计日志表（示例）
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建审计日志索引
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

-- 授予权限（如果使用单独的用户）
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO self_soul_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO self_soul_user;