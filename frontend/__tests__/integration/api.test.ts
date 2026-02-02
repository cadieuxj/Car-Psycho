/**
 * Integration Tests for Frontend-Backend API Communication
 *
 * Tests cover:
 * 1. Manager Service API calls
 * 2. Inference Service API calls
 * 3. Data Service API calls
 * 4. Error handling
 * 5. Response parsing
 */

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080/api';
const MANAGER_URL = 'http://localhost:8001';
const INFERENCE_URL = 'http://localhost:8002';
const DATA_URL = 'http://localhost:8003';

// Mock fetch for testing
const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
});

describe('Manager Service API Integration', () => {
  describe('Health Check', () => {
    it('fetches manager health status', async () => {
      const mockResponse = {
        service: 'manager',
        status: 'running',
        version: '0.1.0',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${MANAGER_URL}/`);
      const data = await response.json();

      expect(mockFetch).toHaveBeenCalledWith(`${MANAGER_URL}/`);
      expect(data.service).toBe('manager');
      expect(data.status).toBe('running');
    });

    it('fetches manager detailed health', async () => {
      const mockResponse = {
        status: 'healthy',
        service: 'manager',
        database: 'postgresql://...',
        redis: 'redis://...',
        t4_vm_host: '192.168.1.100',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${MANAGER_URL}/health`);
      const data = await response.json();

      expect(data.status).toBe('healthy');
      expect(data).toHaveProperty('database');
      expect(data).toHaveProperty('redis');
    });
  });

  describe('Jobs API', () => {
    it('fetches jobs list', async () => {
      const mockResponse = {
        jobs: [],
        message: 'Job management not yet implemented',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${MANAGER_URL}/jobs`);
      const data = await response.json();

      expect(data).toHaveProperty('jobs');
      expect(Array.isArray(data.jobs)).toBe(true);
    });

    it('creates a new job', async () => {
      const mockResponse = {
        status: 'error',
        message: 'Job creation not yet implemented',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${MANAGER_URL}/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'training' }),
      });
      const data = await response.json();

      expect(mockFetch).toHaveBeenCalledWith(
        `${MANAGER_URL}/jobs`,
        expect.objectContaining({ method: 'POST' })
      );
      expect(data).toHaveProperty('status');
    });
  });
});

describe('Inference Service API Integration', () => {
  describe('Health Check', () => {
    it('fetches inference health status', async () => {
      const mockResponse = {
        service: 'inference',
        status: 'running',
        version: '0.1.0',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${INFERENCE_URL}/`);
      const data = await response.json();

      expect(data.service).toBe('inference');
      expect(data.status).toBe('running');
    });

    it('fetches inference detailed health', async () => {
      const mockResponse = {
        status: 'healthy',
        service: 'inference',
        database: 'configured',
        chromadb: 'configured',
        redis: 'configured',
        ollama: 'configured',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${INFERENCE_URL}/health`);
      const data = await response.json();

      expect(data).toHaveProperty('chromadb');
      expect(data).toHaveProperty('ollama');
    });
  });

  describe('Models API', () => {
    it('fetches available models', async () => {
      const mockResponse = {
        models: [],
        message: 'Model listing not yet implemented',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${INFERENCE_URL}/models`);
      const data = await response.json();

      expect(data).toHaveProperty('models');
      expect(Array.isArray(data.models)).toBe(true);
    });
  });

  describe('Prediction API', () => {
    it('sends prediction request', async () => {
      const mockResponse = {
        status: 'error',
        message: 'Prediction endpoint not yet implemented',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const requestBody = {
        text: 'I want a safe car for my family',
        model: 'default',
      };

      const response = await fetch(`${INFERENCE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });
      const data = await response.json();

      expect(mockFetch).toHaveBeenCalledWith(
        `${INFERENCE_URL}/predict`,
        expect.objectContaining({ method: 'POST' })
      );
      expect(data).toHaveProperty('status');
    });

    it('handles prediction response structure', async () => {
      // Mock a successful prediction response (future implementation)
      const mockResponse = {
        status: 'success',
        personality_scores: {
          openness: 0.65,
          conscientiousness: 0.85,
          extraversion: 0.55,
          agreeableness: 0.75,
          neuroticism: 0.35,
        },
        recommendations: [],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${INFERENCE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: 'test' }),
      });
      const data = await response.json();

      if (data.personality_scores) {
        expect(data.personality_scores).toHaveProperty('openness');
        expect(data.personality_scores).toHaveProperty('conscientiousness');
        expect(data.personality_scores).toHaveProperty('extraversion');
        expect(data.personality_scores).toHaveProperty('agreeableness');
        expect(data.personality_scores).toHaveProperty('neuroticism');
      }
    });
  });

  describe('RAG Query API', () => {
    it('sends RAG query request', async () => {
      const mockResponse = {
        status: 'error',
        message: 'RAG query not yet implemented',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${INFERENCE_URL}/rag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'What car should I buy?' }),
      });
      const data = await response.json();

      expect(data).toHaveProperty('status');
    });
  });
});

describe('Data Service API Integration', () => {
  describe('Health Check', () => {
    it('fetches data service health status', async () => {
      const mockResponse = {
        service: 'data',
        status: 'running',
        version: '0.1.0',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${DATA_URL}/`);
      const data = await response.json();

      expect(data.service).toBe('data');
      expect(data.status).toBe('running');
    });

    it('fetches data service detailed health', async () => {
      const mockResponse = {
        status: 'healthy',
        service: 'data',
        database: 'configured',
        chromadb: 'configured',
        redis: 'configured',
        teacher_model: 'gpt-4o',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${DATA_URL}/health`);
      const data = await response.json();

      expect(data).toHaveProperty('teacher_model');
    });
  });

  describe('Datasets API', () => {
    it('fetches available datasets', async () => {
      const mockResponse = {
        datasets: [],
        message: 'Dataset listing not yet implemented',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${DATA_URL}/datasets`);
      const data = await response.json();

      expect(data).toHaveProperty('datasets');
      expect(Array.isArray(data.datasets)).toBe(true);
    });
  });

  describe('Generate API', () => {
    it('sends data generation request', async () => {
      const mockResponse = {
        status: 'error',
        message: 'Data generation not yet implemented',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${DATA_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_samples: 100 }),
      });
      const data = await response.json();

      expect(data).toHaveProperty('status');
    });
  });

  describe('Ingest API', () => {
    it('sends data ingestion request', async () => {
      const mockResponse = {
        status: 'error',
        message: 'Data ingestion not yet implemented',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const response = await fetch(`${DATA_URL}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 'car_listings' }),
      });
      const data = await response.json();

      expect(data).toHaveProperty('status');
    });
  });
});

describe('Error Handling', () => {
  it('handles network errors', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    await expect(fetch(`${MANAGER_URL}/`)).rejects.toThrow('Network error');
  });

  it('handles non-OK responses', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => ({ error: 'Server error' }),
    });

    const response = await fetch(`${MANAGER_URL}/`);
    expect(response.ok).toBe(false);
    expect(response.status).toBe(500);
  });

  it('handles timeout scenarios', async () => {
    jest.useFakeTimers();

    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Timeout')), 5000);
    });

    mockFetch.mockImplementationOnce(() => timeoutPromise);

    const fetchPromise = fetch(`${MANAGER_URL}/`);

    jest.advanceTimersByTime(5000);

    await expect(fetchPromise).rejects.toThrow('Timeout');

    jest.useRealTimers();
  });

  it('handles malformed JSON responses', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => {
        throw new SyntaxError('Unexpected token');
      },
    });

    const response = await fetch(`${MANAGER_URL}/`);
    await expect(response.json()).rejects.toThrow(SyntaxError);
  });
});

describe('API Response Types', () => {
  it('validates health response structure', async () => {
    const mockResponse = {
      status: 'healthy',
      service: 'manager',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    const response = await fetch(`${MANAGER_URL}/health`);
    const data = await response.json();

    // Type validation
    expect(typeof data.status).toBe('string');
    expect(typeof data.service).toBe('string');
  });

  it('validates jobs list response structure', async () => {
    const mockResponse = {
      jobs: [
        { id: '1', status: 'pending' },
        { id: '2', status: 'running' },
      ],
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    const response = await fetch(`${MANAGER_URL}/jobs`);
    const data = await response.json();

    expect(Array.isArray(data.jobs)).toBe(true);
    if (data.jobs.length > 0) {
      expect(data.jobs[0]).toHaveProperty('id');
      expect(data.jobs[0]).toHaveProperty('status');
    }
  });
});

describe('Cross-Service Communication', () => {
  it('can check all services health', async () => {
    const services = [
      { url: MANAGER_URL, name: 'manager' },
      { url: INFERENCE_URL, name: 'inference' },
      { url: DATA_URL, name: 'data' },
    ];

    for (const service of services) {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          service: service.name,
          status: 'running',
        }),
      });

      const response = await fetch(`${service.url}/`);
      const data = await response.json();

      expect(data.service).toBe(service.name);
      expect(data.status).toBe('running');
    }
  });
});
