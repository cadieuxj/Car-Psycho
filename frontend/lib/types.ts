// OCEAN Personality Traits
export interface OceanScores {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

export interface ConfidenceScores {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

export interface PersonalityProfile {
  scores: OceanScores;
  confidence: ConfidenceScores;
  reasoning?: Record<string, string>;
  timestamp: string;
}

// Service Health
export interface ServiceHealth {
  status: 'healthy' | 'unhealthy' | 'unknown' | 'degraded';
  service: string;
  version?: string;
  database?: unknown;
  chromadb?: unknown;
  redis?: unknown;
  ollama?: unknown;
  t4_vm_host?: string;
  teacher_model?: string;
}

// ============================================================================
// Training Jobs
// ============================================================================

export interface TrainingMetrics {
  loss: number;
  personality_loss: number;
  lm_loss: number;
  ordinal_loss: number;
  mse_loss: number;
  epoch: number;
  total_epochs: number;
  step: number;
  total_steps: number;
  learning_rate: number;
  samples_per_second?: number;
  trait_mae?: {
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
  };
}

export interface EpochMetrics {
  epoch: number;
  train_loss: number;
  personality_loss: number;
  ordinal_loss: number;
  mse_loss: number;
  learning_rate: number;
  trait_mae: {
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
  };
  duration_seconds: number;
}

export interface LossHistoryEntry {
  step: number;
  epoch: number;
  total_loss: number;
  personality_loss: number;
  ordinal_loss: number;
  mse_loss: number;
  learning_rate: number;
}

export interface TrainingJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  model_base: string;
  dataset_id: string;
  config: TrainingConfig;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  current_epoch?: number;
  total_epochs?: number;
  metrics?: TrainingMetrics;
  loss_history?: LossHistoryEntry[];
  epoch_metrics?: EpochMetrics[];
  model_id?: string;
}

export interface TrainingConfig {
  model_base: string;
  dataset_id: string;
  job_name: string;
  ollama_model: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  lm_loss_weight: number;
  personality_loss_weight: number;
  ordinal_loss_weight: number;
  mse_loss_weight: number;
  car_domain_boost: number;
  gradient_accumulation_steps: number;
  max_seq_length: number;
  warmup_steps: number;
  weight_decay: number;
  save_steps: number;
  eval_steps: number;
  input_size: number;
  hidden_size: number;
}

// ============================================================================
// Models
// ============================================================================

export interface Model {
  id: string;
  name: string;
  version: string;
  model_type: string;
  base_model: string;
  description?: string;
  storage_path: string;
  status: 'active' | 'training' | 'archived';
  is_default: boolean;
  is_loaded?: boolean;
  metrics?: {
    final_loss: number;
    personality_loss: number;
    trait_mae: Record<string, number>;
  };
  config?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  training_job_id?: string;
}

// ============================================================================
// Datasets
// ============================================================================

export interface TraitStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  histogram: number[];
}

export interface DatasetStats {
  total_samples: number;
  traits: {
    openness: TraitStats;
    conscientiousness: TraitStats;
    extraversion: TraitStats;
    agreeableness: TraitStats;
    neuroticism: TraitStats;
  };
  source_breakdown: Record<string, number>;
}

export interface Dataset {
  id: string;
  name: string;
  dataset_type: string;
  source?: string;
  description?: string;
  file_path: string;
  format: string;
  num_samples: number;
  is_processed: boolean;
  stats?: DatasetStats;
  created_at: string;
  updated_at: string;
}

export interface DatasetPreview {
  dataset_id: string;
  samples: Record<string, unknown>[];
  total_samples: number;
}

// ============================================================================
// Chat / Sales Assistant
// ============================================================================

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  personality_update?: PersonalityProfile;
  car_recommendations?: CarRecommendation[];
  thought_trace?: ThoughtTrace;
}

export interface ThoughtTrace {
  reasoning_steps: string[];
  personality_signals: string[];
  recommendation_rationale?: string;
}

export interface Conversation {
  id: string;
  customer_id?: string;
  messages: ChatMessage[];
  current_profile?: PersonalityProfile;
  created_at: string;
  updated_at: string;
}

export interface CarRecommendation {
  id: string;
  make: string;
  model: string;
  year: number;
  price: number;
  image_url?: string;
  match_score: number;
  match_reasons: string[];
  personality_alignment: {
    trait: keyof OceanScores;
    alignment: 'high' | 'medium' | 'low';
    reason: string;
  }[];
  features: string[];
}

export interface CustomerProfile {
  id: string;
  name?: string;
  email?: string;
  phone?: string;
  personality_profile?: PersonalityProfile;
  conversation_count: number;
  last_interaction?: string;
  preferred_cars?: CarRecommendation[];
  created_at: string;
  updated_at: string;
}

export interface PredictionRequest {
  text: string;
  conversation_id?: string;
  include_recommendations?: boolean;
}

export interface PredictionResponse {
  personality: OceanScores;
  confidence: Record<string, number>;
  trait_descriptions: Record<string, string>;
}

export interface DataGenerationRequest {
  teacher_model: 'gpt-4o' | 'gpt-4-turbo' | 'claude-3.5-sonnet' | 'claude-opus-4.5';
  total_samples: number;
  regmix_ratios?: {
    public_general: number;
    car_domain: number;
    synthetic: number;
  };
}

export interface DataGenerationResponse {
  status: 'started' | 'completed' | 'failed';
  dataset_id?: string;
  message: string;
}

// ============================================================================
// WebSocket Messages
// ============================================================================

export interface TrainingMetricMessage {
  type: 'metric';
  job_id: string;
  data: TrainingMetrics;
  timestamp: string;
}

export interface TrainingStatusMessage {
  type: 'status';
  job_id: string;
  status: TrainingJob['status'];
  message?: string;
  timestamp: string;
}

export interface EpochCompleteMessage {
  type: 'epoch_complete';
  job_id: string;
  data: EpochMetrics;
  timestamp: string;
}

export type TrainingWSMessage =
  | TrainingMetricMessage
  | TrainingStatusMessage
  | EpochCompleteMessage;

// ============================================================================
// Default Training Config
// ============================================================================

export const DEFAULT_TRAINING_CONFIG: Omit<TrainingConfig, 'dataset_id' | 'job_name'> = {
  model_base: 'personality-mlp',
  ollama_model: 'llama3.2:1b',
  epochs: 10,
  batch_size: 32,
  learning_rate: 0.001,
  lm_loss_weight: 0.6,
  personality_loss_weight: 0.4,
  ordinal_loss_weight: 0.7,
  mse_loss_weight: 0.3,
  car_domain_boost: 1.2,
  gradient_accumulation_steps: 1,
  max_seq_length: 512,
  warmup_steps: 100,
  weight_decay: 0.01,
  save_steps: 500,
  eval_steps: 100,
  input_size: 2048,
  hidden_size: 2048,
};
