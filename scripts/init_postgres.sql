-- PostgreSQL initialization script for Self Soul AGI System
-- Enables pgcrypto extension for database encryption and sets up initial schema

-- Enable pgcrypto extension for encryption functions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create schema for Self Soul system
CREATE SCHEMA IF NOT EXISTS self_soul;

-- Set default schema
SET search_path TO self_soul, public;

-- Create tables with encryption support
-- Model configurations table
CREATE TABLE IF NOT EXISTS model_configs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    model_id TEXT UNIQUE NOT NULL,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    source TEXT DEFAULT 'local',
    api_config TEXT,  -- JSON format, will be encrypted at application level
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Training records table
CREATE TABLE IF NOT EXISTS training_records (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    model_id TEXT NOT NULL,
    dataset_name TEXT,
    training_config TEXT,  -- JSON format
    metrics TEXT,  -- JSON format
    status TEXT DEFAULT 'running',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by TEXT
);

-- System logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    level TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    details TEXT,  -- JSON format
    user_id TEXT,
    ip_address TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    model_id TEXT NOT NULL,
    success_rate REAL DEFAULT 0.0,
    latency REAL DEFAULT 0.0,
    accuracy REAL DEFAULT 0.0,
    collaboration_score REAL DEFAULT 0.0,
    calls INTEGER DEFAULT 0,
    last_collaboration_time REAL,
    optimization_suggestions TEXT,  -- JSON format
    custom_metrics TEXT,  -- JSON format
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Training status table
CREATE TABLE IF NOT EXISTS training_status (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    model_id TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,
    progress REAL DEFAULT 0.0,
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER DEFAULT 0,
    current_loss REAL DEFAULT 0.0,
    current_accuracy REAL DEFAULT 0.0,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    device TEXT,
    job_id TEXT
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_model_configs_model_id ON model_configs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_records_model_id ON training_records(model_id);
CREATE INDEX IF NOT EXISTS idx_training_records_status ON training_records(status);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component);
CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_model_id ON performance_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_training_status_model_id ON training_status(model_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for model_configs updated_at
DROP TRIGGER IF EXISTS update_model_configs_updated_at ON model_configs;
CREATE TRIGGER update_model_configs_updated_at
    BEFORE UPDATE ON model_configs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create audit log table for security (encrypted fields example)
CREATE TABLE IF NOT EXISTS security_audit_logs (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT,
    action TEXT NOT NULL,
    resource TEXT NOT NULL,
    details TEXT,  -- Encrypted sensitive details
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Example of using pgcrypto for encryption at database level
-- Note: In production, keys should be managed securely
-- This is an example of how to use pgcrypto for column-level encryption
COMMENT ON TABLE security_audit_logs IS 'Security audit logs with encrypted sensitive details';
COMMENT ON COLUMN security_audit_logs.details IS 'Encrypted sensitive details using pgp_sym_encrypt/pgp_sym_decrypt';

-- Create a view for model status overview
CREATE OR REPLACE VIEW model_status_overview AS
SELECT 
    mc.model_id,
    mc.model_name,
    mc.model_type,
    mc.is_active,
    ts.status as training_status,
    ts.progress as training_progress,
    pm.success_rate,
    pm.accuracy,
    pm.latency,
    mc.updated_at
FROM model_configs mc
LEFT JOIN training_status ts ON mc.model_id = ts.model_id
LEFT JOIN (
    SELECT model_id, 
           MAX(timestamp) as latest_timestamp,
           AVG(success_rate) as success_rate,
           AVG(accuracy) as accuracy,
           AVG(latency) as latency
    FROM performance_metrics
    GROUP BY model_id
) pm ON mc.model_id = pm.model_id;

-- Output initialization completion message
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL database initialized successfully for Self Soul AGI System';
    RAISE NOTICE 'pgcrypto extension enabled for encryption support';
    RAISE NOTICE 'Schema: self_soul, Tables: 6, Indexes: 8, View: 1';
END $$;