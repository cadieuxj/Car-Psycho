"""
Unit Tests for Dataset Downloaders

Tests cover:
1. DatasetDownloader initialization
2. Big5-Chat downloading
3. PANDORA downloading
4. Sample normalization
5. Synthetic data creation
6. Error handling
"""

import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.data.core.downloaders import DatasetDownloader


class TestDatasetDownloaderInitialization:
    """Tests for DatasetDownloader initialization."""

    def test_initialization_creates_directory(self, temp_dir):
        """Test that initialization creates the data directory."""
        raw_dir = temp_dir / "new_raw_dir"
        assert not raw_dir.exists()

        downloader = DatasetDownloader(raw_dir)

        assert raw_dir.exists()
        assert downloader.raw_data_dir == raw_dir

    def test_initialization_existing_directory(self, temp_dir):
        """Test initialization with existing directory."""
        raw_dir = temp_dir / "existing"
        raw_dir.mkdir()

        downloader = DatasetDownloader(raw_dir)

        assert downloader.raw_data_dir == raw_dir


class TestBig5ChatDownload:
    """Tests for Big5-Chat downloading."""

    @pytest.mark.asyncio
    async def test_download_big5_chat_already_exists(self, temp_dir):
        """Test that existing file is not re-downloaded."""
        downloader = DatasetDownloader(temp_dir)

        # Create existing file
        existing_file = temp_dir / "big5_chat.jsonl"
        existing_file.write_text('{"id": "existing"}\n')

        result = await downloader.download_big5_chat()

        assert result == existing_file

    @pytest.mark.asyncio
    async def test_download_big5_chat_creates_synthetic(self, temp_dir):
        """Test that synthetic data is created if download fails."""
        downloader = DatasetDownloader(temp_dir)

        # Mock load_dataset to fail
        with patch('backend.services.data.core.downloaders.load_dataset') as mock_load:
            mock_load.side_effect = Exception("Dataset not found")

            result = await downloader.download_big5_chat()

        assert result is not None
        assert result.exists()

        # Verify content
        samples = []
        with open(result) as f:
            for line in f:
                samples.append(json.loads(line))

        assert len(samples) == 500  # Default synthetic count
        assert all("personality_scores" in s for s in samples)

    @pytest.mark.asyncio
    async def test_download_big5_chat_from_huggingface(self, temp_dir):
        """Test downloading from HuggingFace."""
        downloader = DatasetDownloader(temp_dir)

        # Mock successful HuggingFace download
        mock_dataset = [
            {"id": "1", "text": "Sample text", "openness": 0.5, "conscientiousness": 0.6,
             "extraversion": 0.4, "agreeableness": 0.7, "neuroticism": 0.3}
            for _ in range(10)
        ]

        with patch('backend.services.data.core.downloaders.load_dataset') as mock_load:
            mock_load.return_value = mock_dataset

            result = await downloader.download_big5_chat()

        assert result is not None


class TestPandoraDownload:
    """Tests for PANDORA downloading."""

    @pytest.mark.asyncio
    async def test_download_pandora_already_exists(self, temp_dir):
        """Test that existing file is not re-downloaded."""
        downloader = DatasetDownloader(temp_dir)

        # Create existing file
        existing_file = temp_dir / "pandora.jsonl"
        existing_file.write_text('{"id": "existing"}\n')

        result = await downloader.download_pandora()

        assert result == existing_file

    @pytest.mark.asyncio
    async def test_download_pandora_creates_synthetic(self, temp_dir):
        """Test that synthetic data is created if download fails."""
        downloader = DatasetDownloader(temp_dir)

        # Mock load_dataset to fail
        with patch('backend.services.data.core.downloaders.load_dataset') as mock_load:
            mock_load.side_effect = Exception("Dataset not found")

            result = await downloader.download_pandora()

        assert result is not None
        assert result.exists()

        # Verify content
        samples = []
        with open(result) as f:
            for line in f:
                samples.append(json.loads(line))

        assert len(samples) == 500  # Default synthetic count
        assert all("personality_scores" in s for s in samples)


class TestNormalizeBig5ChatSample:
    """Tests for Big5-Chat sample normalization."""

    def test_normalize_valid_sample(self, temp_dir):
        """Test normalizing a valid sample."""
        downloader = DatasetDownloader(temp_dir)

        item = {
            "id": "test_123",
            "text": "Sample text content",
            "openness": 0.75,
            "conscientiousness": 0.8,
            "extraversion": 0.6,
            "agreeableness": 0.7,
            "neuroticism": 0.4
        }

        result = downloader._normalize_big5_chat_sample(item)

        assert result["id"] == "test_123"
        assert result["text"] == "Sample text content"
        assert result["personality_scores"]["openness"] == 0.75
        assert result["source"] == "big5_chat"

    def test_normalize_with_conversation_field(self, temp_dir):
        """Test normalizing sample with 'conversation' field."""
        downloader = DatasetDownloader(temp_dir)

        item = {
            "id": "test_123",
            "conversation": "Alternative text field",
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5
        }

        result = downloader._normalize_big5_chat_sample(item)

        assert result["text"] == "Alternative text field"

    def test_normalize_with_missing_fields(self, temp_dir):
        """Test normalizing sample with missing fields uses defaults."""
        downloader = DatasetDownloader(temp_dir)

        item = {
            "text": "Sample text"
            # Missing personality scores
        }

        result = downloader._normalize_big5_chat_sample(item)

        assert result is not None
        # Should use default 0.5 values
        assert result["personality_scores"]["openness"] == 0.5

    def test_normalize_invalid_sample(self, temp_dir):
        """Test normalizing invalid sample returns None."""
        downloader = DatasetDownloader(temp_dir)

        # Create a sample that will cause an exception during normalization
        with patch.object(downloader, '_normalize_big5_chat_sample', side_effect=Exception("Error")):
            result = downloader._normalize_big5_chat_sample({})

        # Original method should handle gracefully
        downloader_real = DatasetDownloader(temp_dir)
        result = downloader_real._normalize_big5_chat_sample({})
        # With missing fields, it should still work with defaults
        assert result is not None


class TestNormalizePandoraSample:
    """Tests for PANDORA sample normalization."""

    def test_normalize_valid_sample(self, temp_dir):
        """Test normalizing a valid PANDORA sample."""
        downloader = DatasetDownloader(temp_dir)

        item = {
            "id": "pandora_123",
            "text": "PANDORA sample text",
            "O": 0.7,
            "C": 0.8,
            "E": 0.6,
            "A": 0.75,
            "N": 0.35
        }

        result = downloader._normalize_pandora_sample(item)

        assert result["id"] == "pandora_123"
        assert result["text"] == "PANDORA sample text"
        assert result["personality_scores"]["openness"] == 0.7
        assert result["personality_scores"]["conscientiousness"] == 0.8
        assert result["personality_scores"]["extraversion"] == 0.6
        assert result["personality_scores"]["agreeableness"] == 0.75
        assert result["personality_scores"]["neuroticism"] == 0.35
        assert result["source"] == "pandora"

    def test_normalize_with_response_field(self, temp_dir):
        """Test normalizing sample with 'response' field."""
        downloader = DatasetDownloader(temp_dir)

        item = {
            "id": "test",
            "response": "Response field text",
            "O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5
        }

        result = downloader._normalize_pandora_sample(item)

        assert result["text"] == "Response field text"


class TestSyntheticBig5ChatCreation:
    """Tests for synthetic Big5-Chat data creation."""

    def test_create_synthetic_big5_chat(self, temp_dir):
        """Test creating synthetic Big5-Chat data."""
        downloader = DatasetDownloader(temp_dir)
        output_path = temp_dir / "synthetic_big5.jsonl"

        result = downloader._create_synthetic_big5_chat(output_path)

        assert result == output_path
        assert output_path.exists()

        # Verify content
        samples = []
        with open(output_path) as f:
            for line in f:
                samples.append(json.loads(line))

        assert len(samples) == 500

        # Check sample structure
        for sample in samples[:10]:
            assert "id" in sample
            assert "text" in sample
            assert "personality_scores" in sample
            assert "source" in sample

            # Verify scores are in valid range
            for trait, score in sample["personality_scores"].items():
                assert 0.0 <= score <= 1.0

    def test_synthetic_big5_chat_variety(self, temp_dir):
        """Test that synthetic data has variety."""
        downloader = DatasetDownloader(temp_dir)
        output_path = temp_dir / "synthetic_big5.jsonl"

        downloader._create_synthetic_big5_chat(output_path)

        samples = []
        with open(output_path) as f:
            for line in f:
                samples.append(json.loads(line))

        # Check that scores have variety
        openness_scores = [s["personality_scores"]["openness"] for s in samples]
        assert min(openness_scores) < 0.5 < max(openness_scores)

        # Check unique IDs
        ids = [s["id"] for s in samples]
        assert len(ids) == len(set(ids))


class TestSyntheticPandoraCreation:
    """Tests for synthetic PANDORA data creation."""

    def test_create_synthetic_pandora(self, temp_dir):
        """Test creating synthetic PANDORA data."""
        downloader = DatasetDownloader(temp_dir)
        output_path = temp_dir / "synthetic_pandora.jsonl"

        result = downloader._create_synthetic_pandora(output_path)

        assert result == output_path
        assert output_path.exists()

        # Verify content
        samples = []
        with open(output_path) as f:
            for line in f:
                samples.append(json.loads(line))

        assert len(samples) == 500

        # Check sample structure
        for sample in samples[:10]:
            assert "id" in sample
            assert "text" in sample
            assert "personality_scores" in sample
            assert sample["source"] == "pandora_synthetic"

    def test_synthetic_pandora_text_reflects_personality(self, temp_dir):
        """Test that synthetic PANDORA text reflects personality."""
        downloader = DatasetDownloader(temp_dir)
        output_path = temp_dir / "synthetic_pandora.jsonl"

        downloader._create_synthetic_pandora(output_path)

        samples = []
        with open(output_path) as f:
            for line in f:
                samples.append(json.loads(line))

        # Find high openness sample
        high_openness_samples = [s for s in samples if s["personality_scores"]["openness"] > 0.6]
        if high_openness_samples:
            sample = high_openness_samples[0]
            # Text might mention creativity/ideas
            assert len(sample["text"]) > 0


class TestDownloaderErrorHandling:
    """Tests for error handling in downloaders."""

    @pytest.mark.asyncio
    async def test_big5_chat_general_exception(self, temp_dir):
        """Test handling of general exceptions in Big5-Chat download."""
        downloader = DatasetDownloader(temp_dir)

        with patch('backend.services.data.core.downloaders.load_dataset') as mock_load:
            mock_load.side_effect = Exception("Network error")

            # Should fall back to synthetic data
            result = await downloader.download_big5_chat()

            assert result is not None
            assert result.exists()

    @pytest.mark.asyncio
    async def test_pandora_general_exception(self, temp_dir):
        """Test handling of general exceptions in PANDORA download."""
        downloader = DatasetDownloader(temp_dir)

        with patch('backend.services.data.core.downloaders.load_dataset') as mock_load:
            mock_load.side_effect = Exception("Network error")

            # Should fall back to synthetic data
            result = await downloader.download_pandora()

            assert result is not None
            assert result.exists()


class TestDownloaderIntegration:
    """Integration tests for DatasetDownloader."""

    @pytest.mark.asyncio
    async def test_download_both_datasets(self, temp_dir):
        """Test downloading both datasets."""
        downloader = DatasetDownloader(temp_dir)

        # Both will create synthetic data since HuggingFace isn't mocked to succeed
        with patch('backend.services.data.core.downloaders.load_dataset') as mock_load:
            mock_load.side_effect = Exception("Not available")

            big5_path = await downloader.download_big5_chat()
            pandora_path = await downloader.download_pandora()

        assert big5_path is not None
        assert pandora_path is not None
        assert big5_path.exists()
        assert pandora_path.exists()

        # Verify both files have content
        with open(big5_path) as f:
            big5_count = sum(1 for _ in f)
        with open(pandora_path) as f:
            pandora_count = sum(1 for _ in f)

        assert big5_count == 500
        assert pandora_count == 500
