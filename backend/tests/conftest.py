"""
Pytest Configuration and Shared Fixtures for Car-Psycho Backend Tests

This module provides:
1. Common fixtures for testing
2. Mock objects for external services
3. Test data generators
4. Configuration for async tests
"""

import pytest
import asyncio
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
import tempfile
import shutil

# Configure pytest-asyncio
pytest_plugins = ['pytest_asyncio']


# ============================================================================
# Event Loop Fixture
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_data_dir(temp_dir):
    """Create temporary data directories."""
    directories = {
        "raw": temp_dir / "raw",
        "processed": temp_dir / "processed",
        "car_questionnaire": temp_dir / "car_questionnaire",
        "synthetic": temp_dir / "synthetic",
    }
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return directories


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_personality_scores():
    """Sample Big Five personality scores."""
    return {
        "openness": 0.75,
        "conscientiousness": 0.85,
        "extraversion": 0.60,
        "agreeableness": 0.70,
        "neuroticism": 0.40
    }


@pytest.fixture
def sample_car_profile_data():
    """Sample car questionnaire data."""
    return {
        "customer_id": "TEST_001",
        "responses": {
            "Budget Range": "$30,000 - $40,000",
            "Primary Use": "Daily commute and family trips",
            "Family Size": "4 people (2 adults, 2 children)",
            "Key Priorities": "Safety ratings, fuel efficiency, reliability",
            "Aesthetic Preference": "Modern but practical design",
            "Technology Interest": "Backup cameras, blind spot monitoring",
            "Environmental Concern": "Considering hybrid options",
            "Maintenance Attitude": "Want good warranty and low maintenance"
        },
        "raw_text": "Looking for a safe, reliable car for my family."
    }


@pytest.fixture
def sample_labeled_result(sample_personality_scores):
    """Sample labeled result from PsychSteer."""
    return {
        "customer_id": "TEST_001",
        "personality_scores": sample_personality_scores,
        "confidence_scores": {
            "openness": 0.80,
            "conscientiousness": 0.90,
            "extraversion": 0.75,
            "agreeableness": 0.85,
            "neuroticism": 0.70
        },
        "reasoning": {
            "openness": "Shows interest in modern technology features.",
            "conscientiousness": "Strong emphasis on safety and reliability.",
            "extraversion": "Moderate social features interest.",
            "agreeableness": "Family-oriented priorities.",
            "neuroticism": "Balanced decision-making approach."
        },
        "teacher_model": "gpt-4o",
        "raw_response": "{}"
    }


@pytest.fixture
def sample_training_samples(sample_personality_scores):
    """Generate sample training data."""
    samples = []
    for i in range(100):
        np.random.seed(i)
        sample = {
            "id": f"sample_{i}",
            "text": f"Sample text content {i}",
            "personality_scores": {
                "openness": np.random.uniform(0.2, 0.9),
                "conscientiousness": np.random.uniform(0.2, 0.9),
                "extraversion": np.random.uniform(0.2, 0.9),
                "agreeableness": np.random.uniform(0.2, 0.9),
                "neuroticism": np.random.uniform(0.2, 0.9)
            },
            "source": "test"
        }
        samples.append(sample)
    return samples


# ============================================================================
# Tensor Fixtures
# ============================================================================

@pytest.fixture
def sample_tensors():
    """Sample tensors for loss function tests."""
    batch_size = 4
    num_traits = 5
    return {
        "predictions": torch.rand(batch_size, num_traits),
        "targets": torch.rand(batch_size, num_traits),
        "trait_weights": torch.ones(num_traits)
    }


@pytest.fixture
def sample_lm_tensors():
    """Sample tensors for language model tests."""
    batch_size = 4
    seq_len = 10
    vocab_size = 1000
    hidden_size = 768
    return {
        "logits": torch.randn(batch_size, seq_len, vocab_size),
        "targets": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "hidden_states": torch.randn(batch_size, seq_len, hidden_size),
        "source_types": torch.tensor([0, 1, 0, 2])  # public, car, public, synthetic
    }


# ============================================================================
# Mock API Client Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "personality_scores": {
            "openness": 0.75,
            "conscientiousness": 0.85,
            "extraversion": 0.60,
            "agreeableness": 0.70,
            "neuroticism": 0.40
        },
        "confidence_scores": {
            "openness": 0.80,
            "conscientiousness": 0.90,
            "extraversion": 0.75,
            "agreeableness": 0.85,
            "neuroticism": 0.70
        },
        "reasoning": {
            "openness": "Test reasoning for openness.",
            "conscientiousness": "Test reasoning for conscientiousness.",
            "extraversion": "Test reasoning for extraversion.",
            "agreeableness": "Test reasoning for agreeableness.",
            "neuroticism": "Test reasoning for neuroticism."
        }
    }


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI async client."""
    mock_client = AsyncMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps(mock_openai_response)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    return mock_client


@pytest.fixture
def mock_anthropic_client(mock_openai_response):
    """Mock Anthropic async client."""
    mock_client = AsyncMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(mock_openai_response)
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return mock_client


# ============================================================================
# FastAPI Test Client Fixtures
# ============================================================================

@pytest.fixture
def manager_test_client():
    """Test client for Manager service."""
    from fastapi.testclient import TestClient
    from backend.services.manager.main import app
    return TestClient(app)


@pytest.fixture
def inference_test_client():
    """Test client for Inference service."""
    from fastapi.testclient import TestClient
    from backend.services.inference.main import app
    return TestClient(app)


@pytest.fixture
def data_test_client():
    """Test client for Data service."""
    from fastapi.testclient import TestClient
    from backend.services.data.main import app
    return TestClient(app)


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def mock_model_config():
    """Mock model configuration."""
    config = MagicMock()
    config.hidden_size = 768
    config.vocab_size = 32000
    return config


@pytest.fixture
def mock_model_outputs():
    """Mock model outputs for trainer tests."""
    batch_size = 4
    seq_len = 10
    vocab_size = 32000
    hidden_size = 768

    outputs = MagicMock()
    outputs.loss = torch.tensor(2.5)
    outputs.logits = torch.randn(batch_size, seq_len, vocab_size)
    outputs.hidden_states = [torch.randn(batch_size, seq_len, hidden_size)]
    return outputs


# ============================================================================
# Data Source Fixtures
# ============================================================================

@pytest.fixture
def sample_data_sources(sample_training_samples):
    """Create sample DataSource objects for RegMix tests."""
    from backend.services.data.core.regmix import DataSource

    np.random.seed(42)

    # Public general samples
    public_samples = []
    for i in range(400):
        sample = {
            "id": f"pub_{i}",
            "text": f"Public sample {i}",
            "personality_scores": {
                "openness": np.random.uniform(0.3, 0.8),
                "conscientiousness": np.random.uniform(0.3, 0.8),
                "extraversion": np.random.uniform(0.3, 0.8),
                "agreeableness": np.random.uniform(0.3, 0.8),
                "neuroticism": np.random.uniform(0.3, 0.8)
            }
        }
        public_samples.append(sample)

    # Car domain samples
    car_samples = []
    for i in range(400):
        sample = {
            "id": f"car_{i}",
            "text": f"Car sample {i}",
            "personality_scores": {
                "openness": np.random.uniform(0.4, 0.9),
                "conscientiousness": np.random.uniform(0.4, 0.9),
                "extraversion": np.random.uniform(0.4, 0.9),
                "agreeableness": np.random.uniform(0.4, 0.9),
                "neuroticism": np.random.uniform(0.2, 0.7)
            }
        }
        car_samples.append(sample)

    # Synthetic samples
    synthetic_samples = []
    for i in range(200):
        sample = {
            "id": f"syn_{i}",
            "text": f"Synthetic sample {i}",
            "personality_scores": {
                "openness": np.random.uniform(0.2, 0.9),
                "conscientiousness": np.random.uniform(0.2, 0.9),
                "extraversion": np.random.uniform(0.2, 0.9),
                "agreeableness": np.random.uniform(0.2, 0.9),
                "neuroticism": np.random.uniform(0.2, 0.9)
            }
        }
        synthetic_samples.append(sample)

    return {
        "public": DataSource("Big5-Chat", "public_general", public_samples, len(public_samples)),
        "car": DataSource("Car-Questionnaire", "car_domain", car_samples, len(car_samples)),
        "synthetic": DataSource("PsychSteer-Labeled", "synthetic", synthetic_samples, len(synthetic_samples))
    }


# ============================================================================
# File Fixtures
# ============================================================================

@pytest.fixture
def sample_questionnaire_file(temp_data_dir, sample_car_profile_data):
    """Create a sample questionnaire JSON file."""
    file_path = temp_data_dir["car_questionnaire"] / "TEST_001.json"
    with open(file_path, 'w') as f:
        json.dump(sample_car_profile_data, f, indent=2)
    return file_path


@pytest.fixture
def sample_jsonl_file(temp_data_dir, sample_training_samples):
    """Create a sample JSONL data file."""
    file_path = temp_data_dir["raw"] / "test_data.jsonl"
    with open(file_path, 'w') as f:
        for sample in sample_training_samples[:50]:
            f.write(json.dumps(sample) + '\n')
    return file_path


# ============================================================================
# Environment Variable Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    env_vars = {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
        "REDIS_URL": "redis://localhost:6379",
        "CHROMA_URL": "http://localhost:8000",
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "TEACHER_MODEL": "gpt-4o",
        "T4_VM_HOST": "test-vm-host",
        "T4_VM_USER": "test-user",
        "OLLAMA_HOST": "http://localhost:11434"
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars
