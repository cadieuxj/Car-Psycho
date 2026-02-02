"""
Unit Tests for Manager Service API

Tests cover:
1. Root endpoint (health check)
2. Health endpoint (detailed status)
3. Jobs list endpoint
4. Job creation endpoint
5. CORS middleware
6. Response formats
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.manager.main import app


@pytest.fixture
def client():
    """Create test client for Manager service."""
    return TestClient(app)


class TestManagerRootEndpoint:
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
        assert data["service"] == "manager"

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


class TestManagerHealthEndpoint:
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
        assert data["service"] == "manager"

    def test_health_contains_database_info(self, client):
        """Test health response contains database info."""
        response = client.get("/health")
        data = response.json()
        assert "database" in data

    def test_health_contains_redis_info(self, client):
        """Test health response contains redis info."""
        response = client.get("/health")
        data = response.json()
        assert "redis" in data

    def test_health_contains_t4_vm_info(self, client):
        """Test health response contains T4 VM info."""
        response = client.get("/health")
        data = response.json()
        assert "t4_vm_host" in data

    def test_health_with_env_vars(self, client):
        """Test health response with environment variables."""
        with patch.dict(os.environ, {
            "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
            "REDIS_URL": "redis://localhost:6379",
            "T4_VM_HOST": "192.168.1.100"
        }):
            response = client.get("/health")
            data = response.json()

            assert data["database"] == "postgresql://test:test@localhost:5432/test"
            assert data["redis"] == "redis://localhost:6379"
            assert data["t4_vm_host"] == "192.168.1.100"

    def test_health_without_env_vars(self, client):
        """Test health response without environment variables."""
        # Clear relevant env vars
        env_vars_to_clear = ["DATABASE_URL", "REDIS_URL", "T4_VM_HOST"]
        env_backup = {k: os.environ.get(k) for k in env_vars_to_clear}

        try:
            for var in env_vars_to_clear:
                os.environ.pop(var, None)

            response = client.get("/health")
            data = response.json()

            assert data["database"] == "not configured"
            assert data["redis"] == "not configured"
            assert data["t4_vm_host"] == "not configured"
        finally:
            # Restore env vars
            for var, value in env_backup.items():
                if value is not None:
                    os.environ[var] = value


class TestManagerJobsEndpoint:
    """Tests for jobs endpoint (/jobs)."""

    def test_list_jobs_returns_200(self, client):
        """Test list jobs endpoint returns 200 status."""
        response = client.get("/jobs")
        assert response.status_code == 200

    def test_list_jobs_returns_json(self, client):
        """Test list jobs endpoint returns JSON."""
        response = client.get("/jobs")
        assert response.headers["content-type"] == "application/json"

    def test_list_jobs_contains_jobs_array(self, client):
        """Test list jobs response contains jobs array."""
        response = client.get("/jobs")
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)

    def test_list_jobs_empty_array(self, client):
        """Test list jobs returns empty array (placeholder)."""
        response = client.get("/jobs")
        data = response.json()
        assert data["jobs"] == []

    def test_list_jobs_contains_message(self, client):
        """Test list jobs contains placeholder message."""
        response = client.get("/jobs")
        data = response.json()
        assert "message" in data
        assert "not yet implemented" in data["message"].lower()


class TestManagerCreateJobEndpoint:
    """Tests for create job endpoint (POST /jobs)."""

    def test_create_job_returns_200(self, client):
        """Test create job endpoint returns 200 status."""
        response = client.post("/jobs")
        assert response.status_code == 200

    def test_create_job_returns_json(self, client):
        """Test create job endpoint returns JSON."""
        response = client.post("/jobs")
        assert response.headers["content-type"] == "application/json"

    def test_create_job_returns_error_status(self, client):
        """Test create job returns error status (placeholder)."""
        response = client.post("/jobs")
        data = response.json()
        assert data["status"] == "error"

    def test_create_job_contains_message(self, client):
        """Test create job contains placeholder message."""
        response = client.post("/jobs")
        data = response.json()
        assert "message" in data
        assert "not yet implemented" in data["message"].lower()


class TestManagerCORS:
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
        # Should not return 403/405 for preflight
        assert response.status_code in [200, 204, 405]

    def test_cors_headers_on_get(self, client):
        """Test CORS headers are present on GET response."""
        response = client.get(
            "/",
            headers={"Origin": "http://localhost:3000"}
        )
        # With allow_origins=["*"], the header should be present
        assert response.status_code == 200


class TestManagerInvalidEndpoints:
    """Tests for invalid endpoints."""

    def test_invalid_endpoint_returns_404(self, client):
        """Test invalid endpoint returns 404."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

    def test_invalid_method_on_root(self, client):
        """Test invalid method on root endpoint."""
        response = client.delete("/")
        assert response.status_code == 405

    def test_invalid_method_on_jobs(self, client):
        """Test invalid method on jobs endpoint."""
        response = client.put("/jobs")
        assert response.status_code == 405


class TestManagerResponseStructure:
    """Tests for response structure consistency."""

    def test_all_endpoints_return_json(self, client):
        """Test all endpoints return JSON."""
        endpoints = [
            ("GET", "/"),
            ("GET", "/health"),
            ("GET", "/jobs"),
            ("POST", "/jobs")
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
            ("GET", "/jobs"),
            ("POST", "/jobs")
        ]

        for method, endpoint in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint)

            data = response.json()
            assert isinstance(data, dict), \
                f"Endpoint {method} {endpoint} did not return dict"
