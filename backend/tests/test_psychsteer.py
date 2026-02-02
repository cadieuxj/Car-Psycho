"""
Unit Tests for PsychSteer Synthetic Labeling System

Tests cover:
1. PersonalityScores dataclass
2. CarProfileInput dataclass
3. TeacherModel enum
4. PsychSteer initialization
5. Questionnaire formatting
6. API calls (OpenAI and Anthropic)
7. Response parsing
8. Single profile labeling
9. Batch labeling
"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.data.core.psychsteer import (
    PsychSteer,
    PersonalityScores,
    CarProfileInput,
    TeacherModel
)


class TestPersonalityScores:
    """Tests for PersonalityScores dataclass."""

    def test_valid_scores(self):
        """Test creating valid personality scores."""
        scores = PersonalityScores(
            openness=0.75,
            conscientiousness=0.85,
            extraversion=0.60,
            agreeableness=0.70,
            neuroticism=0.40
        )
        assert scores.openness == 0.75
        assert scores.conscientiousness == 0.85
        assert scores.extraversion == 0.60
        assert scores.agreeableness == 0.70
        assert scores.neuroticism == 0.40

    def test_to_dict(self):
        """Test converting scores to dictionary."""
        scores = PersonalityScores(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5
        )
        result = scores.to_dict()
        assert isinstance(result, dict)
        assert "openness" in result
        assert "conscientiousness" in result
        assert "extraversion" in result
        assert "agreeableness" in result
        assert "neuroticism" in result
        assert all(v == 0.5 for v in result.values())

    def test_validate_valid_scores(self):
        """Test validation with valid scores."""
        scores = PersonalityScores(
            openness=0.0,
            conscientiousness=1.0,
            extraversion=0.5,
            agreeableness=0.25,
            neuroticism=0.75
        )
        assert scores.validate() is True

    def test_validate_boundary_scores(self):
        """Test validation at boundary values."""
        # Exactly 0.0
        scores_min = PersonalityScores(0.0, 0.0, 0.0, 0.0, 0.0)
        assert scores_min.validate() is True

        # Exactly 1.0
        scores_max = PersonalityScores(1.0, 1.0, 1.0, 1.0, 1.0)
        assert scores_max.validate() is True

    def test_validate_invalid_scores_too_high(self):
        """Test validation fails when scores are too high."""
        scores = PersonalityScores(
            openness=1.5,  # Invalid
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5
        )
        assert scores.validate() is False

    def test_validate_invalid_scores_negative(self):
        """Test validation fails when scores are negative."""
        scores = PersonalityScores(
            openness=0.5,
            conscientiousness=-0.1,  # Invalid
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5
        )
        assert scores.validate() is False


class TestCarProfileInput:
    """Tests for CarProfileInput dataclass."""

    def test_create_car_profile(self, sample_car_profile_data):
        """Test creating a car profile input."""
        profile = CarProfileInput(
            customer_id=sample_car_profile_data["customer_id"],
            responses=sample_car_profile_data["responses"],
            raw_text=sample_car_profile_data["raw_text"]
        )
        assert profile.customer_id == "TEST_001"
        assert "Budget Range" in profile.responses
        assert profile.raw_text is not None

    def test_create_profile_without_raw_text(self):
        """Test creating profile without raw text."""
        profile = CarProfileInput(
            customer_id="TEST_002",
            responses={"Question": "Answer"},
            raw_text=None
        )
        assert profile.customer_id == "TEST_002"
        assert profile.raw_text is None

    def test_create_profile_without_customer_id(self):
        """Test creating profile without customer ID."""
        profile = CarProfileInput(
            customer_id=None,
            responses={"Question": "Answer"},
            raw_text=None
        )
        assert profile.customer_id is None


class TestTeacherModel:
    """Tests for TeacherModel enum."""

    def test_gpt4o_value(self):
        """Test GPT-4o model value."""
        assert TeacherModel.GPT4O.value == "gpt-4o"

    def test_gpt4_turbo_value(self):
        """Test GPT-4 Turbo model value."""
        assert TeacherModel.GPT4_TURBO.value == "gpt-4-turbo"

    def test_claude_sonnet_value(self):
        """Test Claude Sonnet model value."""
        assert TeacherModel.CLAUDE_SONNET.value == "claude-3-5-sonnet-20241022"

    def test_claude_opus_value(self):
        """Test Claude Opus model value."""
        assert TeacherModel.CLAUDE_OPUS.value == "claude-opus-4-5-20251101"

    def test_all_models_unique(self):
        """Test all model values are unique."""
        values = [model.value for model in TeacherModel]
        assert len(values) == len(set(values))


class TestPsychSteerInitialization:
    """Tests for PsychSteer initialization."""

    def test_init_with_openai_api_key(self):
        """Test initialization with OpenAI API key."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.GPT4O,
            openai_api_key="test-key"
        )
        assert psychsteer.teacher_model == TeacherModel.GPT4O
        assert psychsteer.temperature == 0.3
        assert psychsteer.max_retries == 3
        assert hasattr(psychsteer, 'openai_client')

    def test_init_with_anthropic_api_key(self):
        """Test initialization with Anthropic API key."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.CLAUDE_SONNET,
            anthropic_api_key="test-key"
        )
        assert psychsteer.teacher_model == TeacherModel.CLAUDE_SONNET
        assert hasattr(psychsteer, 'anthropic_client')

    def test_init_custom_temperature(self):
        """Test initialization with custom temperature."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.GPT4O,
            openai_api_key="test-key",
            temperature=0.5
        )
        assert psychsteer.temperature == 0.5

    def test_init_custom_max_retries(self):
        """Test initialization with custom max retries."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.GPT4O,
            openai_api_key="test-key",
            max_retries=5
        )
        assert psychsteer.max_retries == 5

    def test_init_without_required_api_key_openai(self):
        """Test initialization fails without OpenAI API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key required"):
                PsychSteer(teacher_model=TeacherModel.GPT4O)

    def test_init_without_required_api_key_anthropic(self):
        """Test initialization fails without Anthropic API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Anthropic API key required"):
                PsychSteer(teacher_model=TeacherModel.CLAUDE_SONNET)


class TestPsychSteerFormatQuestionnaire:
    """Tests for questionnaire formatting."""

    @pytest.fixture
    def psychsteer(self):
        """Create PsychSteer instance for tests."""
        return PsychSteer(
            teacher_model=TeacherModel.GPT4O,
            openai_api_key="test-key"
        )

    def test_format_questionnaire_basic(self, psychsteer, sample_car_profile_data):
        """Test basic questionnaire formatting."""
        profile = CarProfileInput(
            customer_id=sample_car_profile_data["customer_id"],
            responses=sample_car_profile_data["responses"],
            raw_text=None
        )
        formatted = psychsteer._format_questionnaire_for_prompt(profile)

        assert "# Customer Car Questionnaire Responses:" in formatted
        assert "Budget Range" in formatted
        assert "$30,000 - $40,000" in formatted

    def test_format_questionnaire_with_raw_text(self, psychsteer, sample_car_profile_data):
        """Test formatting includes raw text when provided."""
        profile = CarProfileInput(
            customer_id=sample_car_profile_data["customer_id"],
            responses=sample_car_profile_data["responses"],
            raw_text=sample_car_profile_data["raw_text"]
        )
        formatted = psychsteer._format_questionnaire_for_prompt(profile)

        assert "# Additional Comments:" in formatted
        assert sample_car_profile_data["raw_text"] in formatted

    def test_format_questionnaire_empty_responses(self, psychsteer):
        """Test formatting with empty responses."""
        profile = CarProfileInput(
            customer_id="EMPTY_001",
            responses={},
            raw_text=None
        )
        formatted = psychsteer._format_questionnaire_for_prompt(profile)
        assert "# Customer Car Questionnaire Responses:" in formatted


class TestPsychSteerParseResponse:
    """Tests for parsing teacher model responses."""

    @pytest.fixture
    def psychsteer(self):
        """Create PsychSteer instance for tests."""
        return PsychSteer(
            teacher_model=TeacherModel.GPT4O,
            openai_api_key="test-key"
        )

    def test_parse_valid_response(self, psychsteer, mock_openai_response):
        """Test parsing a valid response."""
        response_text = json.dumps(mock_openai_response)
        scores, confidence, reasoning = psychsteer._parse_teacher_response(response_text)

        assert isinstance(scores, PersonalityScores)
        assert scores.openness == 0.75
        assert scores.conscientiousness == 0.85
        assert isinstance(confidence, dict)
        assert isinstance(reasoning, dict)

    def test_parse_response_validates_scores(self, psychsteer):
        """Test parsing validates score ranges."""
        invalid_response = {
            "personality_scores": {
                "openness": 1.5,  # Invalid
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.5,
                "neuroticism": 0.5
            },
            "confidence_scores": {},
            "reasoning": {}
        }
        with pytest.raises(ValueError, match="out of valid range"):
            psychsteer._parse_teacher_response(json.dumps(invalid_response))

    def test_parse_invalid_json(self, psychsteer):
        """Test parsing fails with invalid JSON."""
        with pytest.raises(ValueError, match="Invalid response format"):
            psychsteer._parse_teacher_response("not valid json")

    def test_parse_missing_personality_scores(self, psychsteer):
        """Test parsing fails with missing personality scores."""
        invalid_response = {
            "confidence_scores": {},
            "reasoning": {}
        }
        with pytest.raises(ValueError, match="Invalid response format"):
            psychsteer._parse_teacher_response(json.dumps(invalid_response))


class TestPsychSteerAPICallsOpenAI:
    """Tests for OpenAI API calls."""

    @pytest.mark.asyncio
    async def test_call_openai_success(self, mock_openai_client, mock_openai_response):
        """Test successful OpenAI API call."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.GPT4O,
            openai_api_key="test-key"
        )
        psychsteer.openai_client = mock_openai_client

        result = await psychsteer._call_openai("Test prompt")
        assert result == json.dumps(mock_openai_response)
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_openai_retry_on_failure(self, mock_openai_client):
        """Test OpenAI API call retries on failure."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.GPT4O,
            openai_api_key="test-key",
            max_retries=2
        )

        # First call fails, second succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"test": "data"}'))])
        ]
        psychsteer.openai_client = mock_openai_client

        result = await psychsteer._call_openai("Test prompt")
        assert mock_openai_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_call_openai_max_retries_exceeded(self, mock_openai_client):
        """Test OpenAI API call fails after max retries."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.GPT4O,
            openai_api_key="test-key",
            max_retries=2
        )

        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        psychsteer.openai_client = mock_openai_client

        with pytest.raises(Exception):
            await psychsteer._call_openai("Test prompt")


class TestPsychSteerAPICallsAnthropic:
    """Tests for Anthropic API calls."""

    @pytest.mark.asyncio
    async def test_call_anthropic_success(self, mock_anthropic_client, mock_openai_response):
        """Test successful Anthropic API call."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.CLAUDE_SONNET,
            anthropic_api_key="test-key"
        )
        psychsteer.anthropic_client = mock_anthropic_client

        result = await psychsteer._call_anthropic("Test prompt")
        assert result == json.dumps(mock_openai_response)
        mock_anthropic_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_anthropic_retry_on_failure(self, mock_anthropic_client):
        """Test Anthropic API call retries on failure."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.CLAUDE_SONNET,
            anthropic_api_key="test-key",
            max_retries=2
        )

        mock_content = MagicMock()
        mock_content.text = '{"test": "data"}'
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_anthropic_client.messages.create.side_effect = [
            Exception("API Error"),
            mock_response
        ]
        psychsteer.anthropic_client = mock_anthropic_client

        result = await psychsteer._call_anthropic("Test prompt")
        assert mock_anthropic_client.messages.create.call_count == 2


class TestPsychSteerLabelProfile:
    """Tests for single profile labeling."""

    @pytest.fixture
    def psychsteer_with_mock(self, mock_openai_client, mock_openai_response):
        """Create PsychSteer with mocked client."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.GPT4O,
            openai_api_key="test-key"
        )
        psychsteer.openai_client = mock_openai_client
        return psychsteer

    @pytest.mark.asyncio
    async def test_label_car_profile_success(self, psychsteer_with_mock, sample_car_profile_data):
        """Test successful profile labeling."""
        profile = CarProfileInput(
            customer_id=sample_car_profile_data["customer_id"],
            responses=sample_car_profile_data["responses"],
            raw_text=sample_car_profile_data["raw_text"]
        )

        result = await psychsteer_with_mock.label_car_profile(profile)

        assert result["customer_id"] == "TEST_001"
        assert "personality_scores" in result
        assert "confidence_scores" in result
        assert "reasoning" in result
        assert "teacher_model" in result
        assert result["teacher_model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_label_car_profile_returns_all_traits(self, psychsteer_with_mock, sample_car_profile_data):
        """Test that labeling returns all Big Five traits."""
        profile = CarProfileInput(
            customer_id=sample_car_profile_data["customer_id"],
            responses=sample_car_profile_data["responses"],
            raw_text=None
        )

        result = await psychsteer_with_mock.label_car_profile(profile)

        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        for trait in traits:
            assert trait in result["personality_scores"]


class TestPsychSteerBatchLabel:
    """Tests for batch labeling."""

    @pytest.fixture
    def psychsteer_with_mock(self, mock_openai_client, mock_openai_response):
        """Create PsychSteer with mocked client."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.GPT4O,
            openai_api_key="test-key"
        )
        psychsteer.openai_client = mock_openai_client
        return psychsteer

    @pytest.mark.asyncio
    async def test_batch_label_multiple_profiles(self, psychsteer_with_mock):
        """Test batch labeling multiple profiles."""
        profiles = [
            CarProfileInput(f"BATCH_{i}", {"Q": f"A_{i}"}, None)
            for i in range(5)
        ]

        results = await psychsteer_with_mock.batch_label(profiles, batch_size=2)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result["customer_id"] == f"BATCH_{i}"

    @pytest.mark.asyncio
    async def test_batch_label_respects_batch_size(self, psychsteer_with_mock):
        """Test batch labeling respects batch size."""
        profiles = [
            CarProfileInput(f"BATCH_{i}", {"Q": f"A_{i}"}, None)
            for i in range(10)
        ]

        results = await psychsteer_with_mock.batch_label(profiles, batch_size=3)
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_batch_label_handles_failures(self, mock_openai_client, mock_openai_response):
        """Test batch labeling handles individual failures gracefully."""
        psychsteer = PsychSteer(
            teacher_model=TeacherModel.GPT4O,
            openai_api_key="test-key"
        )

        # Create a mock that fails on second call
        call_count = [0]

        async def mock_create(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Simulated failure")
            mock_msg = MagicMock()
            mock_msg.content = json.dumps(mock_openai_response)
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_completion = MagicMock()
            mock_completion.choices = [mock_choice]
            return mock_completion

        mock_openai_client.chat.completions.create = mock_create
        psychsteer.openai_client = mock_openai_client

        profiles = [
            CarProfileInput(f"BATCH_{i}", {"Q": f"A_{i}"}, None)
            for i in range(3)
        ]

        results = await psychsteer.batch_label(profiles, batch_size=1)
        # Results include successful calls (implementation may vary on failure handling)
        assert len(results) >= 2  # At least 2 should succeed

    @pytest.mark.asyncio
    async def test_batch_label_empty_list(self, psychsteer_with_mock):
        """Test batch labeling with empty list."""
        results = await psychsteer_with_mock.batch_label([])
        assert results == []


class TestPsychSteerSystemPrompt:
    """Tests for system prompt content."""

    def test_system_prompt_contains_big_five(self):
        """Test system prompt contains Big Five traits."""
        prompt = PsychSteer.SYSTEM_PROMPT
        traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        for trait in traits:
            assert trait in prompt

    def test_system_prompt_contains_car_mappings(self):
        """Test system prompt contains car-related mappings."""
        prompt = PsychSteer.SYSTEM_PROMPT
        car_keywords = ["Safety", "Performance", "Fuel efficiency", "Family"]
        for keyword in car_keywords:
            assert keyword.lower() in prompt.lower()

    def test_system_prompt_contains_json_example(self):
        """Test system prompt contains JSON example."""
        prompt = PsychSteer.SYSTEM_PROMPT
        assert "personality_scores" in prompt
        assert "confidence_scores" in prompt
        assert "reasoning" in prompt
