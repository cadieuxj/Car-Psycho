-- Car-Psycho Database Initialization Script
-- PostgreSQL Schema for Psychometric Car Sales Platform

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgcrypto for encryption
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- MODEL REGISTRY
-- ============================================================================

-- Model registry table
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'base', 'fine_tuned', 'quantized'
    base_model VARCHAR(255),
    description TEXT,
    storage_path TEXT NOT NULL,
    config JSONB,
    metrics JSONB, -- validation metrics
    status VARCHAR(50) DEFAULT 'active', -- 'active', 'archived', 'training'
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    UNIQUE(name, version)
);

-- Training jobs table
CREATE TABLE IF NOT EXISTS training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_name VARCHAR(255) NOT NULL,
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    config JSONB NOT NULL, -- axolotl config
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed', 'cancelled'
    progress FLOAT DEFAULT 0.0,
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER,
    loss_history JSONB, -- array of {epoch, step, loss, val_loss}
    metrics JSONB,
    vm_host VARCHAR(255),
    vm_job_id VARCHAR(255),
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DATASET MANAGEMENT
-- ============================================================================

-- Datasets table
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    dataset_type VARCHAR(50) NOT NULL, -- 'public_general', 'car_domain', 'synthetic'
    source VARCHAR(255), -- 'big5-chat', 'pandora', 'questionnaire', 'psychsteer'
    description TEXT,
    file_path TEXT NOT NULL,
    format VARCHAR(50) DEFAULT 'jsonl', -- 'jsonl', 'csv', 'parquet'
    num_samples INTEGER,
    schema_info JSONB,
    stats JSONB, -- statistics about the data
    is_processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data samples table (for storing individual samples)
CREATE TABLE IF NOT EXISTS data_samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    sample_data JSONB NOT NULL,
    personality_labels JSONB, -- Big Five scores
    car_profile JSONB, -- Car needs profile
    source_text TEXT,
    is_synthetic BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- PSYCHOMETRIC PROFILES
-- ============================================================================

-- Customer profiles table
CREATE TABLE IF NOT EXISTS customer_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id VARCHAR(255), -- external customer ID
    session_id VARCHAR(255) NOT NULL,

    -- Big Five personality scores (0.0 to 1.0)
    openness FLOAT CHECK (openness >= 0.0 AND openness <= 1.0),
    conscientiousness FLOAT CHECK (conscientiousness >= 0.0 AND conscientiousness <= 1.0),
    extraversion FLOAT CHECK (extraversion >= 0.0 AND extraversion <= 1.0),
    agreeableness FLOAT CHECK (agreeableness >= 0.0 AND agreeableness <= 1.0),
    neuroticism FLOAT CHECK (neuroticism >= 0.0 AND neuroticism <= 1.0),

    -- Confidence scores
    confidence_scores JSONB,

    -- Car needs profile
    car_profile JSONB, -- {budget, usage, family_size, aesthetic_preference, etc.}

    -- Questionnaire responses
    questionnaire_responses JSONB,

    -- Conversation history
    conversation_history JSONB,

    -- Model used for inference
    model_id UUID REFERENCES models(id),
    model_version VARCHAR(50),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(session_id)
);

-- Profile updates table (for tracking real-time updates)
CREATE TABLE IF NOT EXISTS profile_updates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_id UUID REFERENCES customer_profiles(id) ON DELETE CASCADE,
    update_type VARCHAR(50), -- 'initial', 'incremental', 'final'

    -- Updated scores
    openness FLOAT,
    conscientiousness FLOAT,
    extraversion FLOAT,
    agreeableness FLOAT,
    neuroticism FLOAT,

    -- Context that triggered the update
    trigger_message TEXT,
    thought_trace TEXT, -- Chain of thought from model

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INFERENCE LOGS
-- ============================================================================

-- Inference requests table
CREATE TABLE IF NOT EXISTS inference_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    profile_id UUID REFERENCES customer_profiles(id) ON DELETE SET NULL,

    -- Request details
    request_type VARCHAR(50), -- 'questionnaire', 'chat', 'profile'
    input_text TEXT,

    -- Response
    output_text TEXT,
    personality_scores JSONB,
    car_recommendations JSONB,

    -- Model info
    model_id UUID REFERENCES models(id),
    model_version VARCHAR(50),

    -- Performance metrics
    latency_ms INTEGER,
    tokens_used INTEGER,

    -- RAG details
    retrieved_contexts JSONB,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- REGMIX CONFIGURATION
-- ============================================================================

-- RegMix configurations table
CREATE TABLE IF NOT EXISTS regmix_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_name VARCHAR(255) NOT NULL UNIQUE,

    -- Mixing ratios (must sum to 100)
    public_general_ratio INTEGER CHECK (public_general_ratio >= 0 AND public_general_ratio <= 100),
    car_domain_ratio INTEGER CHECK (car_domain_ratio >= 0 AND car_domain_ratio <= 100),
    synthetic_ratio INTEGER CHECK (synthetic_ratio >= 0 AND synthetic_ratio <= 100),

    -- Dataset IDs
    public_datasets UUID[],
    car_datasets UUID[],
    synthetic_datasets UUID[],

    -- Validation
    is_valid BOOLEAN DEFAULT FALSE,
    validation_message TEXT,

    -- Usage tracking
    used_in_training_jobs UUID[],

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT ratio_sum_check CHECK (
        public_general_ratio + car_domain_ratio + synthetic_ratio = 100
    )
);

-- ============================================================================
-- SYSTEM CONFIGURATION
-- ============================================================================

-- API keys table (encrypted)
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service VARCHAR(50) NOT NULL UNIQUE, -- 'openai', 'anthropic', 'huggingface'
    api_key_encrypted TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System settings table
CREATE TABLE IF NOT EXISTS system_settings (
    key VARCHAR(255) PRIMARY KEY,
    value TEXT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Model registry indexes
CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_type ON models(model_type);
CREATE INDEX idx_models_default ON models(is_default) WHERE is_default = TRUE;

-- Training jobs indexes
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_model ON training_jobs(model_id);
CREATE INDEX idx_training_jobs_created ON training_jobs(created_at DESC);

-- Dataset indexes
CREATE INDEX idx_datasets_type ON datasets(dataset_type);
CREATE INDEX idx_datasets_source ON datasets(source);
CREATE INDEX idx_data_samples_dataset ON data_samples(dataset_id);

-- Profile indexes
CREATE INDEX idx_customer_profiles_session ON customer_profiles(session_id);
CREATE INDEX idx_customer_profiles_customer ON customer_profiles(customer_id);
CREATE INDEX idx_profile_updates_profile ON profile_updates(profile_id);
CREATE INDEX idx_profile_updates_created ON profile_updates(created_at DESC);

-- Inference logs indexes
CREATE INDEX idx_inference_logs_session ON inference_logs(session_id);
CREATE INDEX idx_inference_logs_profile ON inference_logs(profile_id);
CREATE INDEX idx_inference_logs_created ON inference_logs(created_at DESC);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update_updated_at trigger to relevant tables
CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_jobs_updated_at BEFORE UPDATE ON training_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_customer_profiles_updated_at BEFORE UPDATE ON customer_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_regmix_configs_updated_at BEFORE UPDATE ON regmix_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON api_keys
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert default system settings
INSERT INTO system_settings (key, value, description) VALUES
    ('default_model', 'llama3.2:1b', 'Default model for inference'),
    ('teacher_model', 'gpt-4o', 'Teacher model for PsychSteer'),
    ('default_batch_size', '4', 'Default batch size for training'),
    ('default_learning_rate', '2e-4', 'Default learning rate'),
    ('max_sequence_length', '2048', 'Maximum sequence length'),
    ('regmix_public_ratio', '40', 'Default RegMix public data ratio'),
    ('regmix_car_ratio', '40', 'Default RegMix car domain ratio'),
    ('regmix_synthetic_ratio', '20', 'Default RegMix synthetic data ratio')
ON CONFLICT (key) DO NOTHING;

-- Insert default RegMix configuration
INSERT INTO regmix_configs (
    config_name,
    public_general_ratio,
    car_domain_ratio,
    synthetic_ratio,
    is_valid
) VALUES (
    'default',
    40,
    40,
    20,
    TRUE
) ON CONFLICT (config_name) DO NOTHING;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View for active training jobs with model info
CREATE OR REPLACE VIEW active_training_jobs AS
SELECT
    tj.id,
    tj.job_name,
    tj.status,
    tj.progress,
    tj.current_epoch,
    tj.total_epochs,
    tj.started_at,
    m.name AS model_name,
    m.version AS model_version
FROM training_jobs tj
LEFT JOIN models m ON tj.model_id = m.id
WHERE tj.status IN ('pending', 'running')
ORDER BY tj.created_at DESC;

-- View for recent customer profiles with latest scores
CREATE OR REPLACE VIEW recent_profiles AS
SELECT
    cp.id,
    cp.customer_id,
    cp.session_id,
    cp.openness,
    cp.conscientiousness,
    cp.extraversion,
    cp.agreeableness,
    cp.neuroticism,
    cp.car_profile,
    m.name AS model_name,
    m.version AS model_version,
    cp.created_at
FROM customer_profiles cp
LEFT JOIN models m ON cp.model_id = m.id
ORDER BY cp.created_at DESC;

-- ============================================================================
-- GRANTS (adjust for your user)
-- ============================================================================

-- Grant permissions to carpsycho user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO carpsycho;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO carpsycho;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO carpsycho;

-- ============================================================================
-- COMPLETION
-- ============================================================================

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully!';
    RAISE NOTICE 'Schema version: 1.0.0';
    RAISE NOTICE 'Tables created: %', (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public');
END $$;
