import {
  ServiceHealth,
  TrainingJob,
  TrainingConfig,
  Model,
  Dataset,
  PredictionRequest,
  PredictionResponse,
  DataGenerationRequest,
  DataGenerationResponse,
  ChatMessage,
  PersonalityProfile,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `API Error: ${response.status}`);
    }

    return response.json();
  }

  // Health endpoints
  async getManagerHealth(): Promise<ServiceHealth> {
    try {
      return await this.request<ServiceHealth>('/api/training/health');
    } catch {
      return { status: 'unhealthy', service: 'manager' };
    }
  }

  async getInferenceHealth(): Promise<ServiceHealth> {
    try {
      return await this.request<ServiceHealth>('/api/inference/health');
    } catch {
      return { status: 'unhealthy', service: 'inference' };
    }
  }

  async getDataHealth(): Promise<ServiceHealth> {
    try {
      return await this.request<ServiceHealth>('/api/datasets/health');
    } catch {
      return { status: 'unhealthy', service: 'data' };
    }
  }

  async getAllServicesHealth(): Promise<{
    manager: ServiceHealth;
    inference: ServiceHealth;
    data: ServiceHealth;
  }> {
    const [manager, inference, data] = await Promise.all([
      this.getManagerHealth(),
      this.getInferenceHealth(),
      this.getDataHealth(),
    ]);
    return { manager, inference, data };
  }

  // Training Jobs
  async listJobs(): Promise<TrainingJob[]> {
    const response = await this.request<{ jobs: TrainingJob[] }>('/api/training/jobs');
    return response.jobs;
  }

  async createJob(config: TrainingConfig): Promise<TrainingJob> {
    return this.request<TrainingJob>('/api/training/jobs', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async getJob(jobId: string): Promise<TrainingJob> {
    return this.request<TrainingJob>(`/api/training/jobs/${jobId}`);
  }

  async cancelJob(jobId: string): Promise<void> {
    await this.request(`/api/training/jobs/${jobId}/cancel`, {
      method: 'POST',
    });
  }

  // Models
  async listModels(): Promise<Model[]> {
    const response = await this.request<{ models: Model[] }>('/api/inference/models');
    return response.models;
  }

  async getModel(modelId: string): Promise<Model> {
    return this.request<Model>(`/api/inference/models/${modelId}`);
  }

  // Inference
  async predict(request: PredictionRequest): Promise<PredictionResponse> {
    return this.request<PredictionResponse>('/api/inference/predict', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async chat(
    message: string,
    conversationId?: string
  ): Promise<{
    response: string;
    personality_update?: PersonalityProfile;
    thought_trace?: { reasoning_steps: string[]; personality_signals: string[] };
  }> {
    return this.request('/api/inference/chat', {
      method: 'POST',
      body: JSON.stringify({ message, conversation_id: conversationId }),
    });
  }

  // Datasets
  async listDatasets(): Promise<Dataset[]> {
    const response = await this.request<{ datasets: Dataset[] }>('/api/datasets/datasets');
    return response.datasets;
  }

  async generateData(request: DataGenerationRequest): Promise<DataGenerationResponse> {
    return this.request<DataGenerationResponse>('/api/datasets/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async ingestData(source: string): Promise<{ status: string; message: string }> {
    return this.request('/api/datasets/ingest', {
      method: 'POST',
      body: JSON.stringify({ source }),
    });
  }
}

export const api = new ApiClient();
export default ApiClient;
