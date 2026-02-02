"""
Unit Tests for Data Ingestion Pipeline

Tests cover:
1. DataPipelineConfig
2. DataPipeline initialization
3. Directory creation
4. Questionnaire loading
5. Sample questionnaire creation
6. Public dataset loading
7. PsychSteer application
8. RegMix application
9. Full pipeline execution
"""

import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.data.core.data_ingest import (
    DataPipeline,
    DataPipelineConfig
)
from backend.services.data.core.psychsteer import CarProfileInput, TeacherModel
from backend.services.data.core.regmix import DataSource


class TestDataPipelineConfig:
    """Tests for DataPipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataPipelineConfig()

        assert config.raw_data_dir == Path("data/raw")
        assert config.processed_data_dir == Path("data/processed")
        assert config.car_questionnaire_dir == Path("data/car_questionnaire")
        assert config.synthetic_data_dir == Path("data/synthetic")
        assert config.download_big5_chat is True
        assert config.download_pandora is True
        assert config.teacher_model == "gpt-4o"
        assert config.psychsteer_batch_size == 10
        assert config.psychsteer_temperature == 0.3
        assert config.regmix_public_ratio == 40
        assert config.regmix_car_ratio == 40
        assert config.regmix_synthetic_ratio == 20
        assert config.output_format == "jsonl"

    def test_custom_config(self, temp_dir):
        """Test custom configuration values."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            teacher_model="claude-3-5-sonnet-20241022",
            psychsteer_batch_size=5,
            regmix_total_samples=500
        )

        assert config.raw_data_dir == temp_dir / "raw"
        assert config.teacher_model == "claude-3-5-sonnet-20241022"
        assert config.psychsteer_batch_size == 5
        assert config.regmix_total_samples == 500

    def test_config_ratios(self):
        """Test custom ratio configuration."""
        config = DataPipelineConfig(
            regmix_public_ratio=50,
            regmix_car_ratio=30,
            regmix_synthetic_ratio=20
        )

        assert config.regmix_public_ratio == 50
        assert config.regmix_car_ratio == 30
        assert config.regmix_synthetic_ratio == 20


class TestDataPipelineInitialization:
    """Tests for DataPipeline initialization."""

    def test_default_initialization(self, temp_dir):
        """Test default initialization."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output"
        )

        pipeline = DataPipeline(config)

        assert pipeline.config is not None

    def test_creates_directories(self, temp_dir):
        """Test that initialization creates directories."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output"
        )

        pipeline = DataPipeline(config)

        assert config.raw_data_dir.exists()
        assert config.processed_data_dir.exists()
        assert config.car_questionnaire_dir.exists()
        assert config.synthetic_data_dir.exists()
        assert config.output_dir.exists()

    def test_initialization_with_none_config(self):
        """Test initialization with None config uses defaults."""
        # This will create directories in default locations
        # We can't easily test this without side effects
        pass


class TestLoadCarQuestionnaires:
    """Tests for loading car questionnaires."""

    def test_load_questionnaires_from_files(self, temp_dir, sample_car_profile_data):
        """Test loading questionnaires from JSON files."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output"
        )

        pipeline = DataPipeline(config)

        # Create questionnaire files
        for i in range(3):
            data = {
                "customer_id": f"TEST_{i}",
                "responses": {"Q1": f"A1_{i}"},
                "raw_text": f"Sample text {i}"
            }
            file_path = config.car_questionnaire_dir / f"TEST_{i}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f)

        profiles = pipeline.load_car_questionnaires()

        assert len(profiles) == 3
        assert all(isinstance(p, CarProfileInput) for p in profiles)

    def test_load_questionnaires_creates_samples_if_empty(self, temp_dir):
        """Test that sample questionnaires are created if directory is empty."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output"
        )

        pipeline = DataPipeline(config)

        # Ensure directory is empty
        assert len(list(config.car_questionnaire_dir.glob("*.json"))) == 0

        profiles = pipeline.load_car_questionnaires()

        # Should have created sample questionnaires
        assert len(profiles) > 0
        assert len(list(config.car_questionnaire_dir.glob("*.json"))) > 0

    def test_load_handles_invalid_files(self, temp_dir):
        """Test loading handles invalid JSON files gracefully."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output"
        )

        pipeline = DataPipeline(config)

        # Create valid file
        valid_data = {"customer_id": "VALID", "responses": {"Q": "A"}}
        with open(config.car_questionnaire_dir / "valid.json", 'w') as f:
            json.dump(valid_data, f)

        # Create invalid file
        with open(config.car_questionnaire_dir / "invalid.json", 'w') as f:
            f.write("not valid json")

        profiles = pipeline.load_car_questionnaires()

        # Should only load valid file
        assert len(profiles) == 1
        assert profiles[0].customer_id == "VALID"


class TestCreateSampleQuestionnaires:
    """Tests for sample questionnaire creation."""

    def test_create_sample_questionnaires(self, temp_dir):
        """Test sample questionnaire creation."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output"
        )

        pipeline = DataPipeline(config)
        pipeline._create_sample_questionnaires()

        files = list(config.car_questionnaire_dir.glob("*.json"))
        assert len(files) == 3  # 3 sample profiles

        # Verify file contents
        for file_path in files:
            with open(file_path) as f:
                data = json.load(f)
            assert "customer_id" in data
            assert "responses" in data


class TestLoadPublicDatasets:
    """Tests for loading public datasets."""

    def test_load_public_datasets_jsonl(self, temp_dir, sample_training_samples):
        """Test loading JSONL datasets."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output"
        )

        pipeline = DataPipeline(config)

        # Create JSONL file
        jsonl_path = temp_dir / "raw" / "test.jsonl"
        with open(jsonl_path, 'w') as f:
            for sample in sample_training_samples[:10]:
                f.write(json.dumps(sample) + '\n')

        downloaded_paths = {"test_dataset": jsonl_path}
        sources = pipeline.load_public_datasets(downloaded_paths)

        assert len(sources) == 1
        assert isinstance(sources[0], DataSource)
        assert len(sources[0].samples) == 10

    def test_load_public_datasets_json(self, temp_dir, sample_training_samples):
        """Test loading JSON datasets."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output"
        )

        pipeline = DataPipeline(config)

        # Create JSON file
        json_path = temp_dir / "raw" / "test.json"
        with open(json_path, 'w') as f:
            json.dump(sample_training_samples[:10], f)

        downloaded_paths = {"test_dataset": json_path}
        sources = pipeline.load_public_datasets(downloaded_paths)

        assert len(sources) == 1
        assert len(sources[0].samples) == 10

    def test_load_multiple_datasets(self, temp_dir, sample_training_samples):
        """Test loading multiple datasets."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output"
        )

        pipeline = DataPipeline(config)

        # Create two files
        path1 = temp_dir / "raw" / "dataset1.jsonl"
        path2 = temp_dir / "raw" / "dataset2.jsonl"

        with open(path1, 'w') as f:
            for sample in sample_training_samples[:5]:
                f.write(json.dumps(sample) + '\n')

        with open(path2, 'w') as f:
            for sample in sample_training_samples[5:15]:
                f.write(json.dumps(sample) + '\n')

        downloaded_paths = {
            "dataset1": path1,
            "dataset2": path2
        }
        sources = pipeline.load_public_datasets(downloaded_paths)

        assert len(sources) == 2


class TestFormatQuestionnaireAsText:
    """Tests for questionnaire text formatting."""

    def test_format_with_reasoning(self, temp_dir, sample_labeled_result):
        """Test formatting with reasoning."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output"
        )

        pipeline = DataPipeline(config)

        text = pipeline._format_questionnaire_as_text(sample_labeled_result)

        assert "Customer:" in text
        assert len(text) > 50  # Should have substantial content

    def test_format_without_reasoning(self, temp_dir):
        """Test formatting without reasoning."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output"
        )

        pipeline = DataPipeline(config)

        labeled_result = {"customer_id": "TEST", "personality_scores": {}}
        text = pipeline._format_questionnaire_as_text(labeled_result)

        assert "Customer:" in text


class TestApplyRegMix:
    """Tests for RegMix application."""

    def test_apply_regmix(self, temp_dir, sample_data_sources):
        """Test applying RegMix."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output",
            regmix_total_samples=100
        )

        pipeline = DataPipeline(config)

        mixed, stats = pipeline.apply_regmix(
            [sample_data_sources["public"]],
            [sample_data_sources["car"]],
            [sample_data_sources["synthetic"]]
        )

        assert len(mixed) == 100
        assert "total_samples" in stats
        assert "distribution" in stats


class TestDownloadPublicDatasets:
    """Tests for public dataset downloading."""

    @pytest.mark.asyncio
    async def test_download_public_datasets(self, temp_dir):
        """Test downloading public datasets."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output",
            download_big5_chat=True,
            download_pandora=True
        )

        pipeline = DataPipeline(config)

        # Mock the DatasetDownloader
        with patch('backend.services.data.core.data_ingest.DatasetDownloader') as MockDownloader:
            mock_instance = MockDownloader.return_value
            mock_instance.download_big5_chat = AsyncMock(return_value=temp_dir / "raw" / "big5.jsonl")
            mock_instance.download_pandora = AsyncMock(return_value=temp_dir / "raw" / "pandora.jsonl")

            # Create mock files
            (temp_dir / "raw" / "big5.jsonl").touch()
            (temp_dir / "raw" / "pandora.jsonl").touch()

            downloaded = await pipeline.download_public_datasets()

            assert "big5_chat" in downloaded
            assert "pandora" in downloaded

    @pytest.mark.asyncio
    async def test_download_disabled(self, temp_dir):
        """Test with downloads disabled."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output",
            download_big5_chat=False,
            download_pandora=False
        )

        pipeline = DataPipeline(config)

        with patch('backend.services.data.core.data_ingest.DatasetDownloader') as MockDownloader:
            downloaded = await pipeline.download_public_datasets()

            assert len(downloaded) == 0


class TestApplyPsychSteerLabeling:
    """Tests for PsychSteer labeling application."""

    @pytest.mark.asyncio
    async def test_apply_psychsteer_labeling(self, temp_dir, mock_openai_response):
        """Test applying PsychSteer labeling."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output",
            teacher_model="gpt-4o"
        )

        pipeline = DataPipeline(config)

        profiles = [
            CarProfileInput("TEST_1", {"Q": "A1"}, None),
            CarProfileInput("TEST_2", {"Q": "A2"}, None)
        ]

        # Mock PsychSteer
        with patch('backend.services.data.core.data_ingest.PsychSteer') as MockPsychSteer:
            mock_instance = MockPsychSteer.return_value
            mock_instance.batch_label = AsyncMock(return_value=[
                {
                    "customer_id": "TEST_1",
                    "personality_scores": mock_openai_response["personality_scores"],
                    "confidence_scores": mock_openai_response["confidence_scores"],
                    "reasoning": mock_openai_response["reasoning"],
                    "teacher_model": "gpt-4o"
                },
                {
                    "customer_id": "TEST_2",
                    "personality_scores": mock_openai_response["personality_scores"],
                    "confidence_scores": mock_openai_response["confidence_scores"],
                    "reasoning": mock_openai_response["reasoning"],
                    "teacher_model": "gpt-4o"
                }
            ])

            samples = await pipeline.apply_psychsteer_labeling(profiles)

            assert len(samples) == 2
            assert samples[0]["id"] == "TEST_1"
            assert "personality_scores" in samples[0]

            # Check file was saved
            output_file = config.synthetic_data_dir / "psychsteer_labeled.jsonl"
            assert output_file.exists()


class TestFullPipeline:
    """Integration tests for the full pipeline."""

    @pytest.mark.asyncio
    async def test_run_full_pipeline_structure(self, temp_dir):
        """Test that full pipeline runs and returns correct structure."""
        config = DataPipelineConfig(
            raw_data_dir=temp_dir / "raw",
            processed_data_dir=temp_dir / "processed",
            car_questionnaire_dir=temp_dir / "car_questionnaire",
            synthetic_data_dir=temp_dir / "synthetic",
            output_dir=temp_dir / "output",
            download_big5_chat=False,
            download_pandora=False,
            regmix_total_samples=10
        )

        pipeline = DataPipeline(config)

        # Create some sample data
        (temp_dir / "raw").mkdir(exist_ok=True)
        sample_data = [
            {"id": f"sample_{i}", "text": f"Text {i}", "personality_scores": {
                "openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
                "agreeableness": 0.5, "neuroticism": 0.5
            }}
            for i in range(50)
        ]
        with open(temp_dir / "raw" / "public.jsonl", 'w') as f:
            for s in sample_data:
                f.write(json.dumps(s) + '\n')

        # Mock the async download function
        async def mock_download():
            return {"public": temp_dir / "raw" / "public.jsonl"}

        with patch.object(pipeline, 'download_public_datasets', mock_download):
            with patch.object(pipeline, 'apply_psychsteer_labeling', AsyncMock(return_value=[])):
                results = await pipeline.run_full_pipeline()

        assert "status" in results
        assert "duration_seconds" in results
        assert "public_datasets" in results
        assert "car_profiles" in results
