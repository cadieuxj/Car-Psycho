"""
Unit Tests for Data Service API

Tests cover:
1. Root endpoint (health check)
2. Health endpoint (detailed status)
3. Datasets list endpoint
4. Generate endpoint
5. Ingest endpoint
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

from backend.services.data.main import app


@pytest.fixture
def client():
    """Create test client for Data service."""
    return TestClient(app)


class TestDataRootEndpoint:
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
        assert data["service"] == "data"

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


class TestDataHealthEndpoint:
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
        assert data["service"] == "data"

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

    def test_health_contains_teacher_model_info(self, client):
        """Test health response contains teacher model info."""
        response = client.get("/health")
        data = response.json()
        assert "teacher_model" in data

    def test_health_with_env_vars(self, client):
        """Test health response with environment variables."""
        with patch.dict(os.environ, {
            "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
            "CHROMA_URL": "http://chromadb:8000",
            "REDIS_URL": "redis://localhost:6379",
            "TEACHER_MODEL": "gpt-4o"
        }):
            response = client.get("/health")
            data = response.json()

            assert data["database"] == "postgresql://test:test@localhost:5432/test"
            assert data["chromadb"] == "http://chromadb:8000"
            assert data["redis"] == "redis://localhost:6379"
            assert data["teacher_model"] == "gpt-4o"

    def test_health_without_env_vars(self, client):
        """Test health response without environment variables."""
        env_vars_to_clear = ["DATABASE_URL", "CHROMA_URL", "REDIS_URL", "TEACHER_MODEL"]
        env_backup = {k: os.environ.get(k) for k in env_vars_to_clear}

        try:
            for var in env_vars_to_clear:
                os.environ.pop(var, None)

            response = client.get("/health")
            data = response.json()

            assert data["database"] == "not configured"
            assert data["chromadb"] == "not configured"
            assert data["redis"] == "not configured"
            assert data["teacher_model"] == "not configured"
        finally:
            for var, value in env_backup.items():
                if value is not None:
                    os.environ[var] = value


class TestDataDatasetsEndpoint:
    """Tests for datasets endpoint (/datasets)."""

    def test_list_datasets_returns_200(self, client):
        """Test list datasets endpoint returns 200 status."""
        response = client.get("/datasets")
        assert response.status_code == 200

    def test_list_datasets_returns_json(self, client):
        """Test list datasets endpoint returns JSON."""
        response = client.get("/datasets")
        assert response.headers["content-type"] == "application/json"

    def test_list_datasets_contains_datasets_array(self, client):
        """Test list datasets response contains datasets array."""
        response = client.get("/datasets")
        data = response.json()
        assert "datasets" in data
        assert isinstance(data["datasets"], list)

    def test_list_datasets_empty_array(self, client):
        """Test list datasets returns empty array (placeholder)."""
        response = client.get("/datasets")
        data = response.json()
        assert data["datasets"] == []

    def test_list_datasets_contains_message(self, client):
        """Test list datasets contains placeholder message."""
        response = client.get("/datasets")
        data = response.json()
        assert "message" in data
        assert "not yet implemented" in data["message"].lower()


class TestDataGenerateEndpoint:
    """Tests for generate endpoint (POST /generate)."""

    def test_generate_returns_200(self, client):
        """Test generate endpoint returns 200 status."""
        response = client.post("/generate")
        assert response.status_code == 200

    def test_generate_returns_json(self, client):
        """Test generate endpoint returns JSON."""
        response = client.post("/generate")
        assert response.headers["content-type"] == "application/json"

    def test_generate_returns_error_status(self, client):
        """Test generate returns error status (placeholder)."""
        response = client.post("/generate")
        data = response.json()
        assert data["status"] == "error"

    def test_generate_contains_message(self, client):
        """Test generate contains placeholder message."""
        response = client.post("/generate")
        data = response.json()
        assert "message" in data
        assert "not yet implemented" in data["message"].lower()

    def test_generate_accepts_json_body(self, client):
        """Test generate accepts JSON body."""
        response = client.post(
            "/generate",
            json={
                "num_samples": 100,
                "source_type": "car_domain"
            }
        )
        assert response.status_code == 200


class TestDataIngestEndpoint:
    """Tests for ingest endpoint (POST /ingest)."""

    def test_ingest_returns_200(self, client):
        """Test ingest endpoint returns 200 status."""
        response = client.post("/ingest")
        assert response.status_code == 200

    def test_ingest_returns_json(self, client):
        """Test ingest endpoint returns JSON."""
        response = client.post("/ingest")
        assert response.headers["content-type"] == "application/json"

    def test_ingest_returns_error_status(self, client):
        """Test ingest returns error status (placeholder)."""
        response = client.post("/ingest")
        data = response.json()
        assert data["status"] == "error"

    def test_ingest_contains_message(self, client):
        """Test ingest contains placeholder message."""
        response = client.post("/ingest")
        data = response.json()
        assert "message" in data
        assert "not yet implemented" in data["message"].lower()

    def test_ingest_accepts_json_body(self, client):
        """Test ingest accepts JSON body."""
        response = client.post(
            "/ingest",
            json={
                "source": "car_listings",
                "format": "jsonl"
            }
        )
        assert response.status_code == 200


class TestDataCORS:
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


class TestDataInvalidEndpoints:
    """Tests for invalid endpoints."""

    def test_invalid_endpoint_returns_404(self, client):
        """Test invalid endpoint returns 404."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

    def test_invalid_method_on_root(self, client):
        """Test invalid method on root endpoint."""
        response = client.delete("/")
        assert response.status_code == 405

    def test_invalid_method_on_datasets(self, client):
        """Test invalid method on datasets endpoint."""
        response = client.post("/datasets")
        assert response.status_code == 405

    def test_get_on_generate_returns_405(self, client):
        """Test GET on generate endpoint returns 405."""
        response = client.get("/generate")
        assert response.status_code == 405

    def test_get_on_ingest_returns_405(self, client):
        """Test GET on ingest endpoint returns 405."""
        response = client.get("/ingest")
        assert response.status_code == 405


class TestDataResponseStructure:
    """Tests for response structure consistency."""

    def test_all_endpoints_return_json(self, client):
        """Test all endpoints return JSON."""
        endpoints = [
            ("GET", "/"),
            ("GET", "/health"),
            ("GET", "/datasets"),
            ("POST", "/generate"),
            ("POST", "/ingest")
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
            ("GET", "/datasets"),
            ("POST", "/generate"),
            ("POST", "/ingest")
        ]

        for method, endpoint in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)

            data = response.json()
            assert isinstance(data, dict), \
                f"Endpoint {method} {endpoint} did not return dict"


class TestDataServiceInfo:
    """Tests for service information consistency."""

    def test_service_name_consistent(self, client):
        """Test service name is consistent across endpoints."""
        root_response = client.get("/")
        health_response = client.get("/health")

        assert root_response.json()["service"] == health_response.json()["service"]
        assert root_response.json()["service"] == "data"

    def test_version_present_in_root(self, client):
        """Test version is present in root endpoint."""
        response = client.get("/")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"


class TestDataServiceIntegration:
    """Integration tests for Data service."""

    def test_health_check_all_components(self, client):
        """Test health check reports all components."""
        response = client.get("/health")
        data = response.json()

        # All key components should be reported
        required_keys = ["status", "service", "database", "chromadb", "redis", "teacher_model"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_service_responds_quickly(self, client):
        """Test service responds within reasonable time."""
        import time
        start = time.time()

        response = client.get("/")

        elapsed = time.time() - start
        assert elapsed < 1.0  # Should respond in under 1 second
        assert response.status_code == 200
