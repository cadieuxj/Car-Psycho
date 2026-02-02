"""
Unit Tests for RegMix Data Mixing System

Tests cover:
1. DataSource dataclass
2. RegMixConfig dataclass and validation
3. RegMix initialization
4. Trait quintile calculation
5. Sample stratification
6. Stratified sampling
7. Dataset mixing
8. Statistics calculation
9. Dataset saving
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.data.core.regmix import (
    RegMix,
    RegMixConfig,
    DataSource
)


class TestDataSource:
    """Tests for DataSource dataclass."""

    def test_create_data_source(self):
        """Test creating a data source."""
        samples = [{"id": i, "text": f"Sample {i}"} for i in range(100)]
        source = DataSource(
            name="TestSource",
            source_type="public_general",
            samples=samples,
            num_samples=100
        )
        assert source.name == "TestSource"
        assert source.source_type == "public_general"
        assert len(source.samples) == 100

    def test_data_source_post_init(self):
        """Test that post_init sets num_samples correctly."""
        samples = [{"id": i} for i in range(50)]
        # Even if we pass wrong num_samples, post_init should correct it
        source = DataSource(
            name="Test",
            source_type="car_domain",
            samples=samples,
            num_samples=0  # Will be overwritten
        )
        assert source.num_samples == 50

    def test_empty_data_source(self):
        """Test creating an empty data source."""
        source = DataSource(
            name="Empty",
            source_type="synthetic",
            samples=[],
            num_samples=0
        )
        assert source.num_samples == 0
        assert len(source.samples) == 0


class TestRegMixConfig:
    """Tests for RegMixConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RegMixConfig()
        assert config.public_general_ratio == 40
        assert config.car_domain_ratio == 40
        assert config.synthetic_ratio == 20
        assert config.total_samples is None
        assert config.stratify_by_traits is True
        assert config.num_quintiles == 5
        assert config.random_seed == 42

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RegMixConfig(
            public_general_ratio=50,
            car_domain_ratio=30,
            synthetic_ratio=20,
            total_samples=1000,
            random_seed=123
        )
        assert config.public_general_ratio == 50
        assert config.car_domain_ratio == 30
        assert config.synthetic_ratio == 20
        assert config.total_samples == 1000
        assert config.random_seed == 123

    def test_validate_valid_ratios(self):
        """Test validation passes with valid ratios."""
        config = RegMixConfig(
            public_general_ratio=40,
            car_domain_ratio=40,
            synthetic_ratio=20
        )
        config.validate()  # Should not raise

    def test_validate_invalid_ratios_sum_less_than_100(self):
        """Test validation fails when ratios sum < 100."""
        config = RegMixConfig(
            public_general_ratio=30,
            car_domain_ratio=30,
            synthetic_ratio=30
        )
        with pytest.raises(ValueError, match="Ratios must sum to 100"):
            config.validate()

    def test_validate_invalid_ratios_sum_greater_than_100(self):
        """Test validation fails when ratios sum > 100."""
        config = RegMixConfig(
            public_general_ratio=50,
            car_domain_ratio=50,
            synthetic_ratio=20
        )
        with pytest.raises(ValueError, match="Ratios must sum to 100"):
            config.validate()

    def test_various_valid_ratio_combinations(self):
        """Test various valid ratio combinations."""
        valid_configs = [
            (33, 33, 34),
            (0, 0, 100),
            (100, 0, 0),
            (50, 50, 0),
            (10, 10, 80),
        ]
        for public, car, synthetic in valid_configs:
            config = RegMixConfig(
                public_general_ratio=public,
                car_domain_ratio=car,
                synthetic_ratio=synthetic
            )
            config.validate()  # Should not raise


class TestRegMixInitialization:
    """Tests for RegMix initialization."""

    def test_default_initialization(self):
        """Test initialization with default config."""
        regmix = RegMix()
        assert regmix.config is not None
        assert regmix.config.public_general_ratio == 40

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = RegMixConfig(
            public_general_ratio=50,
            car_domain_ratio=30,
            synthetic_ratio=20
        )
        regmix = RegMix(config)
        assert regmix.config.public_general_ratio == 50

    def test_initialization_validates_config(self):
        """Test that initialization validates config."""
        invalid_config = RegMixConfig(
            public_general_ratio=50,
            car_domain_ratio=50,
            synthetic_ratio=50
        )
        with pytest.raises(ValueError):
            RegMix(invalid_config)

    def test_initialization_sets_random_seed(self):
        """Test that initialization sets random seed."""
        config = RegMixConfig(random_seed=12345)
        regmix = RegMix(config)
        # Create two regmix instances with same seed should produce same results
        regmix1 = RegMix(RegMixConfig(random_seed=42))
        regmix2 = RegMix(RegMixConfig(random_seed=42))
        # Both should be initialized correctly
        assert regmix1.config.random_seed == regmix2.config.random_seed


class TestRegMixTraitQuintile:
    """Tests for trait quintile calculation."""

    @pytest.fixture
    def regmix(self):
        """Create RegMix instance for tests."""
        return RegMix()

    def test_quintile_low_scores(self, regmix):
        """Test quintile calculation for low scores."""
        scores = {"openness": 0.05, "conscientiousness": 0.1}
        quintiles = regmix._get_trait_quintile(scores)
        assert quintiles["openness"] == 0
        assert quintiles["conscientiousness"] == 0

    def test_quintile_mid_scores(self, regmix):
        """Test quintile calculation for mid scores."""
        scores = {"openness": 0.5}
        quintiles = regmix._get_trait_quintile(scores)
        assert quintiles["openness"] == 2

    def test_quintile_high_scores(self, regmix):
        """Test quintile calculation for high scores."""
        scores = {"openness": 0.95}
        quintiles = regmix._get_trait_quintile(scores)
        assert quintiles["openness"] == 4

    def test_quintile_boundary_scores(self, regmix):
        """Test quintile calculation at boundaries."""
        # 0.0 should be quintile 0
        assert regmix._get_trait_quintile({"o": 0.0})["o"] == 0
        # 0.19 should be quintile 0
        assert regmix._get_trait_quintile({"o": 0.19})["o"] == 0
        # 0.2 should be quintile 1
        assert regmix._get_trait_quintile({"o": 0.2})["o"] == 1
        # 1.0 should be capped at quintile 4
        assert regmix._get_trait_quintile({"o": 1.0})["o"] == 4

    def test_quintile_all_traits(self, regmix, sample_personality_scores):
        """Test quintile calculation for all Big Five traits."""
        quintiles = regmix._get_trait_quintile(sample_personality_scores)
        assert len(quintiles) == 5
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            assert trait in quintiles
            assert 0 <= quintiles[trait] <= 4


class TestRegMixStratification:
    """Tests for sample stratification."""

    @pytest.fixture
    def regmix(self):
        """Create RegMix instance for tests."""
        return RegMix()

    @pytest.fixture
    def diverse_samples(self):
        """Create samples with diverse personality profiles."""
        samples = []
        np.random.seed(42)
        for i in range(100):
            sample = {
                "id": f"sample_{i}",
                "personality_scores": {
                    "openness": np.random.uniform(0.1, 0.9),
                    "conscientiousness": np.random.uniform(0.1, 0.9),
                    "extraversion": np.random.uniform(0.1, 0.9),
                    "agreeableness": np.random.uniform(0.1, 0.9),
                    "neuroticism": np.random.uniform(0.1, 0.9)
                }
            }
            samples.append(sample)
        return samples

    def test_stratify_creates_strata(self, regmix, diverse_samples):
        """Test that stratification creates strata."""
        strata = regmix._stratify_samples(diverse_samples)
        assert len(strata) > 0
        # All strata should have samples
        for stratum_samples in strata.values():
            assert len(stratum_samples) > 0

    def test_stratify_preserves_all_samples(self, regmix, diverse_samples):
        """Test that stratification preserves all samples."""
        strata = regmix._stratify_samples(diverse_samples)
        total_samples = sum(len(s) for s in strata.values())
        assert total_samples == len(diverse_samples)

    def test_stratify_missing_personality_scores(self, regmix):
        """Test stratification handles missing personality scores."""
        samples = [
            {"id": "1", "personality_scores": {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5}},
            {"id": "2"},  # Missing personality_scores
        ]
        strata = regmix._stratify_samples(samples)
        # Only valid samples should be stratified
        total = sum(len(s) for s in strata.values())
        assert total == 1


class TestRegMixStratifiedSampling:
    """Tests for stratified sampling."""

    @pytest.fixture
    def regmix(self):
        """Create RegMix instance for tests."""
        return RegMix()

    @pytest.fixture
    def regmix_no_stratify(self):
        """Create RegMix instance without stratification."""
        config = RegMixConfig(stratify_by_traits=False)
        return RegMix(config)

    @pytest.fixture
    def diverse_samples(self):
        """Create diverse sample data."""
        samples = []
        np.random.seed(42)
        for i in range(100):
            sample = {
                "id": f"sample_{i}",
                "personality_scores": {
                    "openness": np.random.uniform(0.1, 0.9),
                    "conscientiousness": np.random.uniform(0.1, 0.9),
                    "extraversion": np.random.uniform(0.1, 0.9),
                    "agreeableness": np.random.uniform(0.1, 0.9),
                    "neuroticism": np.random.uniform(0.1, 0.9)
                }
            }
            samples.append(sample)
        return samples

    def test_stratified_sample_correct_count(self, regmix, diverse_samples):
        """Test stratified sampling returns correct number of samples."""
        sampled = regmix._stratified_sample(diverse_samples, 50)
        assert len(sampled) == 50

    def test_stratified_sample_respects_maximum(self, regmix, diverse_samples):
        """Test stratified sampling doesn't exceed available samples."""
        sampled = regmix._stratified_sample(diverse_samples, 200)
        assert len(sampled) == len(diverse_samples)

    def test_simple_random_sampling(self, regmix_no_stratify, diverse_samples):
        """Test simple random sampling without stratification."""
        sampled = regmix_no_stratify._stratified_sample(diverse_samples, 30)
        assert len(sampled) == 30

    def test_stratified_sample_empty_input(self, regmix):
        """Test stratified sampling with empty input."""
        sampled = regmix._stratified_sample([], 10)
        assert len(sampled) == 0


class TestRegMixDatasetMixing:
    """Tests for the main mix_datasets function."""

    @pytest.fixture
    def regmix(self):
        """Create RegMix instance for tests."""
        config = RegMixConfig(total_samples=100)
        return RegMix(config)

    def test_mix_datasets_basic(self, regmix, sample_data_sources):
        """Test basic dataset mixing."""
        mixed, stats = regmix.mix_datasets(
            [sample_data_sources["public"]],
            [sample_data_sources["car"]],
            [sample_data_sources["synthetic"]]
        )
        assert len(mixed) == 100
        assert "total_samples" in stats
        assert "distribution" in stats

    def test_mix_datasets_correct_ratios(self, sample_data_sources):
        """Test dataset mixing respects ratios."""
        config = RegMixConfig(
            public_general_ratio=40,
            car_domain_ratio=40,
            synthetic_ratio=20,
            total_samples=100
        )
        regmix = RegMix(config)

        mixed, stats = regmix.mix_datasets(
            [sample_data_sources["public"]],
            [sample_data_sources["car"]],
            [sample_data_sources["synthetic"]]
        )

        # Check distribution percentages (with some tolerance)
        public_pct = stats["distribution"]["public_general"]["percentage"]
        car_pct = stats["distribution"]["car_domain"]["percentage"]
        synthetic_pct = stats["distribution"]["synthetic"]["percentage"]

        assert 35 <= public_pct <= 45  # Around 40%
        assert 35 <= car_pct <= 45  # Around 40%
        assert 15 <= synthetic_pct <= 25  # Around 20%

    def test_mix_datasets_adds_source_type(self, regmix, sample_data_sources):
        """Test that mixing adds _source_type to samples."""
        mixed, _ = regmix.mix_datasets(
            [sample_data_sources["public"]],
            [sample_data_sources["car"]],
            [sample_data_sources["synthetic"]]
        )

        for sample in mixed:
            assert "_source_type" in sample
            assert sample["_source_type"] in ["public_general", "car_domain", "synthetic"]

    def test_mix_datasets_shuffles_output(self, sample_data_sources):
        """Test that output is shuffled."""
        config = RegMixConfig(total_samples=100, random_seed=42)
        regmix = RegMix(config)

        mixed, _ = regmix.mix_datasets(
            [sample_data_sources["public"]],
            [sample_data_sources["car"]],
            [sample_data_sources["synthetic"]]
        )

        # Check that samples are not grouped by source type
        source_types = [s["_source_type"] for s in mixed[:10]]
        # Should have some variety in first 10 samples
        assert len(set(source_types)) > 1

    def test_mix_datasets_calculates_trait_distributions(self, regmix, sample_data_sources):
        """Test that trait distributions are calculated."""
        _, stats = regmix.mix_datasets(
            [sample_data_sources["public"]],
            [sample_data_sources["car"]],
            [sample_data_sources["synthetic"]]
        )

        assert "trait_distributions" in stats
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            assert trait in stats["trait_distributions"]
            assert "mean" in stats["trait_distributions"][trait]
            assert "std" in stats["trait_distributions"][trait]
            assert "min" in stats["trait_distributions"][trait]
            assert "max" in stats["trait_distributions"][trait]


class TestRegMixStatisticsCalculation:
    """Tests for statistics calculation."""

    @pytest.fixture
    def regmix(self):
        """Create RegMix instance for tests."""
        return RegMix()

    def test_calculate_statistics(self, regmix, sample_training_samples):
        """Test statistics calculation."""
        public = sample_training_samples[:40]
        car = sample_training_samples[40:80]
        synthetic = sample_training_samples[80:]

        stats = regmix._calculate_statistics(
            sample_training_samples,
            public,
            car,
            synthetic
        )

        assert stats["total_samples"] == len(sample_training_samples)
        assert stats["distribution"]["public_general"]["count"] == 40
        assert stats["distribution"]["car_domain"]["count"] == 40
        assert stats["distribution"]["synthetic"]["count"] == 20

    def test_trait_distributions(self, regmix, sample_training_samples):
        """Test trait distribution calculation."""
        distributions = regmix._get_trait_distributions(sample_training_samples)

        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            assert trait in distributions
            # Mean should be between 0 and 1
            assert 0 <= distributions[trait]["mean"] <= 1
            # Std should be positive
            assert distributions[trait]["std"] >= 0


class TestRegMixSaveDataset:
    """Tests for saving mixed datasets."""

    @pytest.fixture
    def regmix(self):
        """Create RegMix instance for tests."""
        return RegMix()

    def test_save_as_jsonl(self, regmix, sample_training_samples, temp_dir):
        """Test saving dataset as JSONL."""
        output_path = temp_dir / "output.jsonl"
        regmix.save_mixed_dataset(sample_training_samples, output_path, format="jsonl")

        assert output_path.exists()

        # Verify content
        loaded = []
        with open(output_path, 'r') as f:
            for line in f:
                loaded.append(json.loads(line))
        assert len(loaded) == len(sample_training_samples)

    def test_save_as_json(self, regmix, sample_training_samples, temp_dir):
        """Test saving dataset as JSON."""
        output_path = temp_dir / "output.json"
        regmix.save_mixed_dataset(sample_training_samples, output_path, format="json")

        assert output_path.exists()

        # Verify content
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        assert len(loaded) == len(sample_training_samples)

    def test_save_invalid_format(self, regmix, sample_training_samples, temp_dir):
        """Test saving with invalid format raises error."""
        output_path = temp_dir / "output.txt"
        with pytest.raises(ValueError, match="Unsupported format"):
            regmix.save_mixed_dataset(sample_training_samples, output_path, format="txt")

    def test_save_creates_parent_directories(self, regmix, sample_training_samples, temp_dir):
        """Test saving creates parent directories if needed."""
        output_path = temp_dir / "nested" / "dir" / "output.jsonl"
        regmix.save_mixed_dataset(sample_training_samples, output_path, format="jsonl")
        assert output_path.exists()


class TestRegMixEdgeCases:
    """Tests for edge cases and error handling."""

    def test_mix_with_empty_sources(self):
        """Test mixing with some empty sources."""
        config = RegMixConfig(
            public_general_ratio=100,
            car_domain_ratio=0,
            synthetic_ratio=0,
            total_samples=50
        )
        regmix = RegMix(config)

        np.random.seed(42)
        public_samples = [{
            "id": f"pub_{i}",
            "personality_scores": {
                "openness": np.random.uniform(0.3, 0.8),
                "conscientiousness": np.random.uniform(0.3, 0.8),
                "extraversion": np.random.uniform(0.3, 0.8),
                "agreeableness": np.random.uniform(0.3, 0.8),
                "neuroticism": np.random.uniform(0.3, 0.8)
            }
        } for i in range(100)]

        public_source = DataSource("Public", "public_general", public_samples, len(public_samples))
        empty_car = DataSource("Car", "car_domain", [], 0)
        empty_synthetic = DataSource("Synthetic", "synthetic", [], 0)

        mixed, stats = regmix.mix_datasets([public_source], [empty_car], [empty_synthetic])
        assert len(mixed) == 50
        assert all(s["_source_type"] == "public_general" for s in mixed)

    def test_mix_multiple_sources_same_type(self, sample_data_sources):
        """Test mixing with multiple sources of the same type."""
        config = RegMixConfig(total_samples=100)
        regmix = RegMix(config)

        # Create second public source
        np.random.seed(123)
        public_samples_2 = [{
            "id": f"pub2_{i}",
            "personality_scores": {
                "openness": np.random.uniform(0.3, 0.8),
                "conscientiousness": np.random.uniform(0.3, 0.8),
                "extraversion": np.random.uniform(0.3, 0.8),
                "agreeableness": np.random.uniform(0.3, 0.8),
                "neuroticism": np.random.uniform(0.3, 0.8)
            }
        } for i in range(100)]
        public_source_2 = DataSource("Big5-Chat-2", "public_general", public_samples_2, len(public_samples_2))

        mixed, stats = regmix.mix_datasets(
            [sample_data_sources["public"], public_source_2],
            [sample_data_sources["car"]],
            [sample_data_sources["synthetic"]]
        )
        assert len(mixed) == 100
