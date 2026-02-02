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
  status: 'healthy' | 'unhealthy' | 'unknown';
  service: string;
  version?: string;
  database?: string;
  chromadb?: string;
  redis?: string;
  ollama?: string;
  t4_vm_host?: string;
  teacher_model?: string;
}

// Training Jobs
export interface TrainingJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  model_base: string;
  dataset_id: string;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  metrics?: TrainingMetrics;
}

export interface TrainingMetrics {
  loss: number;
  personality_loss: number;
  lm_loss: number;
  epoch: number;
  step: number;
  learning_rate: number;
}

export interface TrainingConfig {
  model_base: string;
  dataset_id: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  lm_loss_weight: number;
  personality_loss_weight: number;
  ordinal_loss_weight: number;
  mse_loss_weight: number;
  car_domain_boost: number;
}

// Models
export interface Model {
  id: string;
  name: string;
  version: string;
  base_model: string;
  status: 'ready' | 'training' | 'failed';
  metrics?: {
    mae: number;
    rmse: number;
    r2: number;
  };
  created_at: string;
  training_job_id?: string;
}

// Datasets
export interface Dataset {
  id: string;
  name: string;
  description?: string;
  total_samples: number;
  public_general_samples: number;
  car_domain_samples: number;
  synthetic_samples: number;
  created_at: string;
  teacher_model?: string;
}

// Chat / Sales Assistant
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

// Car Recommendations
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

// Customer Profiles
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

// Prediction Request/Response
export interface PredictionRequest {
  text: string;
  conversation_id?: string;
  include_recommendations?: boolean;
}

export interface PredictionResponse {
  personality: PersonalityProfile;
  recommendations?: CarRecommendation[];
  thought_trace?: ThoughtTrace;
}

// Data Generation
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
