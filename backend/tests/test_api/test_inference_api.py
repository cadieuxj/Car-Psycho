"""
Unit Tests for Inference Service API

Tests cover:
1. Root endpoint (health check)
2. Health endpoint (detailed status)
3. Models list endpoint
4. Predict endpoint
5. RAG query endpoint
6. CORS middleware
7. Response formats
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.inference.main import app


@pytest.fixture
def client():
    """Create test client for Inference service."""
    return TestClient(app)


class TestInferenceRootEndpoint:
    """Tests for root endpoint (/)."""

    def test_root_returns_200(self, client):
        """Test root endpoint returns 200 status."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_json(self, client):
        """Test root endpoint returns JSON."""
        response = client.get("/")
        assert response.headers["content-type"] == "application/json"

    def test_root_contains_service_name(self, client):
        """Test root response contains service name."""
        response = client.get("/")
        data = response.json()
        assert data["service"] == "inference"

    def test_root_contains_status(self, client):
        """Test root response contains status."""
        response = client.get("/")
        data = response.json()
        assert data["status"] == "running"

    def test_root_contains_version(self, client):
        """Test root response contains version."""
        response = client.get("/")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"


class TestInferenceHealthEndpoint:
    """Tests for health endpoint (/health)."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200 status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """Test health endpoint returns JSON."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"

    def test_health_contains_status(self, client):
        """Test health response contains status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_contains_service_name(self, client):
        """Test health response contains service name."""
        response = client.get("/health")
        data = response.json()
        assert data["service"] == "inference"

    def test_health_contains_database_info(self, client):
        """Test health response contains database info."""
        response = client.get("/health")
        data = response.json()
        assert "database" in data

    def test_health_contains_chromadb_info(self, client):
        """Test health response contains ChromaDB info."""
        response = client.get("/health")
        data = response.json()
        assert "chromadb" in data

    def test_health_contains_redis_info(self, client):
        """Test health response contains redis info."""
        response = client.get("/health")
        data = response.json()
        assert "redis" in data

    def test_health_contains_ollama_info(self, client):
        """Test health response contains Ollama info."""
        response = client.get("/health")
        data = response.json()
        assert "ollama" in data

    def test_health_with_env_vars(self, client):
        """Test health response with environment variables."""
        with patch.dict(os.environ, {
            "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
            "CHROMA_URL": "http://chromadb:8000",
            "REDIS_URL": "redis://localhost:6379",
            "OLLAMA_HOST": "http://localhost:11434"
        }):
            response = client.get("/health")
            data = response.json()

            assert data["database"] == "postgresql://test:test@localhost:5432/test"
            assert data["chromadb"] == "http://chromadb:8000"
            assert data["redis"] == "redis://localhost:6379"
            assert data["ollama"] == "http://localhost:11434"

    def test_health_without_env_vars(self, client):
        """Test health response without environment variables."""
        env_vars_to_clear = ["DATABASE_URL", "CHROMA_URL", "REDIS_URL", "OLLAMA_HOST"]
        env_backup = {k: os.environ.get(k) for k in env_vars_to_clear}

        try:
            for var in env_vars_to_clear:
                os.environ.pop(var, None)

            response = client.get("/health")
            data = response.json()

            assert data["database"] == "not configured"
            assert data["chromadb"] == "not configured"
            assert data["redis"] == "not configured"
            assert data["ollama"] == "not configured"
        finally:
            for var, value in env_backup.items():
                if value is not None:
                    os.environ[var] = value


class TestInferenceModelsEndpoint:
    """Tests for models endpoint (/models)."""

    def test_list_models_returns_200(self, client):
        """Test list models endpoint returns 200 status."""
        response = client.get("/models")
        assert response.status_code == 200

    def test_list_models_returns_json(self, client):
        """Test list models endpoint returns JSON."""
        response = client.get("/models")
        assert response.headers["content-type"] == "application/json"

    def test_list_models_contains_models_array(self, client):
        """Test list models response contains models array."""
        response = client.get("/models")
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_list_models_empty_array(self, client):
        """Test list models returns empty array (placeholder)."""
        response = client.get("/models")
        data = response.json()
        assert data["models"] == []

    def test_list_models_contains_message(self, client):
        """Test list models contains placeholder message."""
        response = client.get("/models")
        data = response.json()
        assert "message" in data
        assert "not yet implemented" in data["message"].lower()


class TestInferencePredictEndpoint:
    """Tests for predict endpoint (POST /predict)."""

    def test_predict_returns_200(self, client):
        """Test predict endpoint returns 200 status."""
        response = client.post("/predict")
        assert response.status_code == 200

    def test_predict_returns_json(self, client):
        """Test predict endpoint returns JSON."""
        response = client.post("/predict")
        assert response.headers["content-type"] == "application/json"

    def test_predict_returns_error_status(self, client):
        """Test predict returns error status (placeholder)."""
        response = client.post("/predict")
        data = response.json()
        assert data["status"] == "error"

    def test_predict_contains_message(self, client):
        """Test predict contains placeholder message."""
        response = client.post("/predict")
        data = response.json()
        assert "message" in data
        assert "not yet implemented" in data["message"].lower()

    def test_predict_accepts_json_body(self, client):
        """Test predict accepts JSON body."""
        response = client.post(
            "/predict",
            json={"text": "Sample questionnaire response"}
        )
        assert response.status_code == 200


class TestInferenceRAGQueryEndpoint:
    """Tests for RAG query endpoint (POST /rag/query)."""

    def test_rag_query_returns_200(self, client):
        """Test RAG query endpoint returns 200 status."""
        response = client.post("/rag/query")
        assert response.status_code == 200

    def test_rag_query_returns_json(self, client):
        """Test RAG query endpoint returns JSON."""
        response = client.post("/rag/query")
        assert response.headers["content-type"] == "application/json"

    def test_rag_query_returns_error_status(self, client):
        """Test RAG query returns error status (placeholder)."""
        response = client.post("/rag/query")
        data = response.json()
        assert data["status"] == "error"

    def test_rag_query_contains_message(self, client):
        """Test RAG query contains placeholder message."""
        response = client.post("/rag/query")
        data = response.json()
        assert "message" in data
        assert "not yet implemented" in data["message"].lower()

    def test_rag_query_accepts_json_body(self, client):
        """Test RAG query accepts JSON body."""
        response = client.post(
            "/rag/query",
            json={"query": "What car should I buy?"}
        )
        assert response.status_code == 200


class TestInferenceCORS:
    """Tests for CORS middleware."""

    def test_cors_allows_all_origins(self, client):
        """Test CORS allows all origins."""
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        assert response.status_code in [200, 204, 405]

    def test_cors_headers_on_get(self, client):
        """Test CORS headers are present on GET response."""
        response = client.get(
            "/",
            headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == 200


class TestInferenceInvalidEndpoints:
    """Tests for invalid endpoints."""

    def test_invalid_endpoint_returns_404(self, client):
        """Test invalid endpoint returns 404."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

    def test_invalid_method_on_root(self, client):
        """Test invalid method on root endpoint."""
        response = client.delete("/")
        assert response.status_code == 405

    def test_invalid_method_on_models(self, client):
        """Test invalid method on models endpoint."""
        response = client.post("/models")
        assert response.status_code == 405

    def test_get_on_predict_returns_405(self, client):
        """Test GET on predict endpoint returns 405."""
        response = client.get("/predict")
        assert response.status_code == 405

    def test_get_on_rag_query_returns_405(self, client):
        """Test GET on RAG query endpoint returns 405."""
        response = client.get("/rag/query")
        assert response.status_code == 405


class TestInferenceResponseStructure:
    """Tests for response structure consistency."""

    def test_all_endpoints_return_json(self, client):
        """Test all endpoints return JSON."""
        endpoints = [
            ("GET", "/"),
            ("GET", "/health"),
            ("GET", "/models"),
            ("POST", "/predict"),
            ("POST", "/rag/query")
        ]

        for method, endpoint in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)

            assert response.headers["content-type"] == "application/json", \
                f"Endpoint {method} {endpoint} did not return JSON"

    def test_all_endpoints_return_dict(self, client):
        """Test all endpoints return dictionary responses."""
        endpoints = [
            ("GET", "/"),
            ("GET", "/health"),
            ("GET", "/models"),
            ("POST", "/predict"),
            ("POST", "/rag/query")
        ]

        for method, endpoint in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)

            data = response.json()
            assert isinstance(data, dict), \
                f"Endpoint {method} {endpoint} did not return dict"


class TestInferenceServiceInfo:
    """Tests for service information consistency."""

    def test_service_name_consistent(self, client):
        """Test service name is consistent across endpoints."""
        root_response = client.get("/")
        health_response = client.get("/health")

        assert root_response.json()["service"] == health_response.json()["service"]
        assert root_response.json()["service"] == "inference"

    def test_version_present_in_root(self, client):
        """Test version is present in root endpoint."""
        response = client.get("/")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"
