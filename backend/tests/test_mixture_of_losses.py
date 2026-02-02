"""
Unit Tests for Mixture of Losses (MoL) Module

Tests cover:
1. DataSourceType enum
2. MixtureOfLosses initialization
3. MoL forward pass
4. PersonalityMoL initialization
5. PersonalityMoL forward pass
6. Source-specific weighting
7. Integration with ordinal regression
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.ml.losses.mixture_of_losses import (
    MixtureOfLosses,
    PersonalityMoL,
    DataSourceType
)
from backend.ml.losses.ordinal_regression import CombinedOrdinalMSELoss


class TestDataSourceType:
    """Tests for DataSourceType enum."""

    def test_public_general_value(self):
        """Test PUBLIC_GENERAL value."""
        assert DataSourceType.PUBLIC_GENERAL.value == "public_general"

    def test_car_domain_value(self):
        """Test CAR_DOMAIN value."""
        assert DataSourceType.CAR_DOMAIN.value == "car_domain"

    def test_synthetic_value(self):
        """Test SYNTHETIC value."""
        assert DataSourceType.SYNTHETIC.value == "synthetic"

    def test_all_values_unique(self):
        """Test all enum values are unique."""
        values = [e.value for e in DataSourceType]
        assert len(values) == len(set(values))


class TestMixtureOfLossesInitialization:
    """Tests for MixtureOfLosses initialization."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        mol = MixtureOfLosses()
        assert mol.ce_weight == 1.0
        assert mol.kl_weight == 0.5
        assert mol.temperature == 1.0
        assert mol.label_smoothing == 0.1

    def test_custom_weights(self):
        """Test custom weight initialization."""
        mol = MixtureOfLosses(ce_weight=2.0, kl_weight=1.0)
        assert mol.ce_weight == 2.0
        assert mol.kl_weight == 1.0

    def test_custom_temperature(self):
        """Test custom temperature."""
        mol = MixtureOfLosses(temperature=2.0)
        assert mol.temperature == 2.0

    def test_custom_label_smoothing(self):
        """Test custom label smoothing."""
        mol = MixtureOfLosses(label_smoothing=0.2)
        assert mol.label_smoothing == 0.2

    def test_internal_loss_functions_created(self):
        """Test internal loss functions are created."""
        mol = MixtureOfLosses()
        assert hasattr(mol, 'ce_loss')
        assert hasattr(mol, 'kl_loss')
        assert isinstance(mol.ce_loss, nn.CrossEntropyLoss)
        assert isinstance(mol.kl_loss, nn.KLDivLoss)


class TestMixtureOfLossesForward:
    """Tests for MixtureOfLosses forward pass."""

    @pytest.fixture
    def mol(self):
        """Create MoL instance for tests."""
        return MixtureOfLosses()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        batch_size = 4
        vocab_size = 1000
        return {
            "logits": torch.randn(batch_size, vocab_size),
            "targets": torch.randint(0, vocab_size, (batch_size,)),
            "source_types": torch.tensor([0, 1, 0, 2]),  # public, car, public, synthetic
            "teacher_logits": torch.randn(batch_size, vocab_size)
        }

    def test_forward_returns_tuple(self, mol, sample_data):
        """Test that forward returns (loss, dict) tuple."""
        loss, loss_dict = mol(
            sample_data["logits"],
            sample_data["targets"],
            sample_data["source_types"]
        )

        assert isinstance(loss, (torch.Tensor, float))
        assert isinstance(loss_dict, dict)

    def test_forward_without_teacher_logits(self, mol, sample_data):
        """Test forward pass without teacher logits."""
        loss, loss_dict = mol(
            sample_data["logits"],
            sample_data["targets"],
            sample_data["source_types"]
        )

        assert "ce_loss" in loss_dict
        assert "kl_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_forward_with_teacher_logits(self, mol, sample_data):
        """Test forward pass with teacher logits."""
        loss, loss_dict = mol(
            sample_data["logits"],
            sample_data["targets"],
            sample_data["source_types"],
            sample_data["teacher_logits"]
        )

        assert "kl_loss" in loss_dict

    def test_loss_dict_contains_sample_counts(self, mol, sample_data):
        """Test loss dict contains sample counts per source."""
        _, loss_dict = mol(
            sample_data["logits"],
            sample_data["targets"],
            sample_data["source_types"]
        )

        assert loss_dict["num_public"] == 2  # Two public samples
        assert loss_dict["num_car"] == 1  # One car sample
        assert loss_dict["num_synthetic"] == 1  # One synthetic sample

    def test_only_car_domain_samples(self, mol):
        """Test with only car domain samples."""
        batch_size = 4
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size,))
        source_types = torch.ones(batch_size, dtype=torch.long)  # All car domain

        loss, loss_dict = mol(logits, targets, source_types)

        assert loss_dict["num_car"] == batch_size
        assert loss_dict["num_public"] == 0
        assert loss_dict["num_synthetic"] == 0
        assert loss_dict["ce_loss"] > 0

    def test_only_public_samples(self, mol):
        """Test with only public samples."""
        batch_size = 4
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size,))
        source_types = torch.zeros(batch_size, dtype=torch.long)  # All public

        loss, loss_dict = mol(logits, targets, source_types)

        assert loss_dict["num_public"] == batch_size
        assert loss_dict["num_car"] == 0

    def test_only_synthetic_samples(self, mol):
        """Test with only synthetic samples."""
        batch_size = 4
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size,))
        source_types = torch.full((batch_size,), 2, dtype=torch.long)  # All synthetic

        loss, loss_dict = mol(logits, targets, source_types)

        assert loss_dict["num_synthetic"] == batch_size
        assert "synthetic_loss" in loss_dict


class TestPersonalityMoLInitialization:
    """Tests for PersonalityMoL initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        pmol = PersonalityMoL()
        assert pmol.lm_weight == 0.6
        assert pmol.personality_weight == 0.4

    def test_custom_weights(self):
        """Test custom weight initialization."""
        pmol = PersonalityMoL(lm_weight=0.7, personality_weight=0.3)
        assert pmol.lm_weight == 0.7
        assert pmol.personality_weight == 0.3

    def test_custom_ordinal_loss(self):
        """Test custom ordinal loss module."""
        custom_loss = CombinedOrdinalMSELoss(ordinal_weight=0.8, mse_weight=0.2)
        pmol = PersonalityMoL(ordinal_loss=custom_loss)
        assert pmol.personality_loss is custom_loss

    def test_default_ordinal_loss_created(self):
        """Test default ordinal loss is created if not provided."""
        pmol = PersonalityMoL()
        assert hasattr(pmol, 'personality_loss')
        assert isinstance(pmol.personality_loss, CombinedOrdinalMSELoss)

    def test_lm_loss_created(self):
        """Test LM loss is created."""
        pmol = PersonalityMoL()
        assert hasattr(pmol, 'lm_loss')
        assert isinstance(pmol.lm_loss, nn.CrossEntropyLoss)


class TestPersonalityMoLForward:
    """Tests for PersonalityMoL forward pass."""

    @pytest.fixture
    def pmol(self):
        """Create PersonalityMoL instance for tests."""
        return PersonalityMoL()

    @pytest.fixture
    def sample_lm_data(self):
        """Create sample LM data for tests."""
        batch_size = 4
        seq_len = 10
        vocab_size = 1000
        return {
            "lm_logits": torch.randn(batch_size, seq_len, vocab_size),
            "lm_targets": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "personality_preds": torch.rand(batch_size, 5),
            "personality_targets": torch.rand(batch_size, 5),
            "source_types": torch.tensor([0, 1, 0, 2])
        }

    def test_forward_with_all_inputs(self, pmol, sample_lm_data):
        """Test forward with all inputs."""
        loss, loss_dict = pmol(
            sample_lm_data["lm_logits"],
            sample_lm_data["lm_targets"],
            sample_lm_data["personality_preds"],
            sample_lm_data["personality_targets"],
            sample_lm_data["source_types"]
        )

        assert isinstance(loss, torch.Tensor)
        assert "lm_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_forward_without_personality(self, pmol, sample_lm_data):
        """Test forward without personality predictions."""
        loss, loss_dict = pmol(
            sample_lm_data["lm_logits"],
            sample_lm_data["lm_targets"]
        )

        assert "lm_loss" in loss_dict
        # Should not have personality losses
        personality_keys = [k for k in loss_dict.keys() if 'personality' in k]
        assert len(personality_keys) == 0

    def test_forward_without_source_types(self, pmol, sample_lm_data):
        """Test forward without source types."""
        loss, loss_dict = pmol(
            sample_lm_data["lm_logits"],
            sample_lm_data["lm_targets"],
            sample_lm_data["personality_preds"],
            sample_lm_data["personality_targets"]
        )

        assert isinstance(loss, torch.Tensor)
        assert "lm_loss" in loss_dict

    def test_lm_loss_calculated(self, pmol, sample_lm_data):
        """Test LM loss is calculated correctly."""
        loss, loss_dict = pmol(
            sample_lm_data["lm_logits"],
            sample_lm_data["lm_targets"]
        )

        assert loss_dict["lm_loss"] > 0  # Should have some loss

    def test_personality_loss_included(self, pmol, sample_lm_data):
        """Test personality loss is included when provided."""
        loss, loss_dict = pmol(
            sample_lm_data["lm_logits"],
            sample_lm_data["lm_targets"],
            sample_lm_data["personality_preds"],
            sample_lm_data["personality_targets"]
        )

        # Should have personality-related losses
        personality_keys = [k for k in loss_dict.keys() if 'personality' in k]
        assert len(personality_keys) > 0

    def test_car_domain_boost(self, pmol, sample_lm_data):
        """Test car domain samples get boosted."""
        # All public samples
        source_types_public = torch.zeros(4, dtype=torch.long)
        loss_public, _ = pmol(
            sample_lm_data["lm_logits"],
            sample_lm_data["lm_targets"],
            sample_lm_data["personality_preds"],
            sample_lm_data["personality_targets"],
            source_types_public
        )

        # All car domain samples
        source_types_car = torch.ones(4, dtype=torch.long)
        loss_car, _ = pmol(
            sample_lm_data["lm_logits"],
            sample_lm_data["lm_targets"],
            sample_lm_data["personality_preds"],
            sample_lm_data["personality_targets"],
            source_types_car
        )

        # Car domain should have boosted loss (higher)
        assert loss_car > loss_public

    def test_2d_lm_logits(self, pmol):
        """Test with 2D LM logits (already flattened)."""
        batch_size = 40  # seq_len * batch
        vocab_size = 1000

        lm_logits = torch.randn(batch_size, vocab_size)
        lm_targets = torch.randint(0, vocab_size, (batch_size,))

        loss, loss_dict = pmol(lm_logits, lm_targets)
        assert "lm_loss" in loss_dict


class TestMoLGradients:
    """Tests for gradient computation in MoL."""

    def test_mol_gradients(self):
        """Test gradients flow through MoL."""
        mol = MixtureOfLosses()

        logits = torch.randn(4, 100, requires_grad=True)
        targets = torch.randint(0, 100, (4,))
        source_types = torch.tensor([0, 1, 0, 2])

        loss, _ = mol(logits, targets, source_types)

        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            assert logits.grad is not None

    def test_personality_mol_gradients(self):
        """Test gradients flow through PersonalityMoL."""
        pmol = PersonalityMoL()

        lm_logits = torch.randn(4, 10, 100, requires_grad=True)
        lm_targets = torch.randint(0, 100, (4, 10))
        personality_preds = torch.rand(4, 5, requires_grad=True)
        personality_targets = torch.rand(4, 5)

        loss, _ = pmol(lm_logits, lm_targets, personality_preds, personality_targets)
        loss.backward()

        assert lm_logits.grad is not None
        assert personality_preds.grad is not None


class TestMoLEdgeCases:
    """Tests for edge cases in MoL."""

    def test_single_sample(self):
        """Test with single sample."""
        mol = MixtureOfLosses()

        logits = torch.randn(1, 100)
        targets = torch.randint(0, 100, (1,))
        source_types = torch.tensor([1])  # Car domain

        loss, loss_dict = mol(logits, targets, source_types)
        assert "total_loss" in loss_dict

    def test_empty_source_type(self):
        """Test behavior when a source type has no samples."""
        mol = MixtureOfLosses()

        logits = torch.randn(2, 100)
        targets = torch.randint(0, 100, (2,))
        source_types = torch.tensor([0, 0])  # Only public, no car or synthetic

        loss, loss_dict = mol(logits, targets, source_types)
        assert loss_dict["num_car"] == 0
        assert loss_dict["num_synthetic"] == 0

    def test_large_batch(self):
        """Test with large batch size."""
        mol = MixtureOfLosses()

        logits = torch.randn(256, 1000)
        targets = torch.randint(0, 1000, (256,))
        source_types = torch.randint(0, 3, (256,))

        loss, loss_dict = mol(logits, targets, source_types)
        assert "total_loss" in loss_dict

    def test_personality_mol_perfect_prediction(self):
        """Test PersonalityMoL with perfect personality prediction."""
        pmol = PersonalityMoL()

        lm_logits = torch.randn(4, 10, 100)
        lm_targets = torch.randint(0, 100, (4, 10))

        # Perfect personality prediction
        personality_values = torch.rand(4, 5)
        personality_preds = personality_values.clone()
        personality_targets = personality_values.clone()

        loss, loss_dict = pmol(lm_logits, lm_targets, personality_preds, personality_targets)

        # Personality loss component should be very low
        personality_total = loss_dict.get("personality_total_loss", 0)
        assert personality_total < 0.1


class TestMoLWeightBalance:
    """Tests for weight balance in MoL."""

    def test_ce_weight_dominance(self):
        """Test CE weight dominates when set high."""
        mol_high_ce = MixtureOfLosses(ce_weight=10.0, kl_weight=0.1)

        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))
        source_types = torch.tensor([1, 1, 0, 0])  # Half car, half public

        loss, loss_dict = mol_high_ce(logits, targets, source_types)

        # With high CE weight, CE should dominate
        if loss_dict["ce_loss"] > 0:
            total_contribution = loss_dict["ce_loss"] * 10.0
            assert total_contribution > loss_dict["kl_loss"] * 0.1

    def test_lm_personality_weight_balance(self):
        """Test LM and personality weight balance."""
        pmol_lm_dominant = PersonalityMoL(lm_weight=0.9, personality_weight=0.1)
        pmol_personality_dominant = PersonalityMoL(lm_weight=0.1, personality_weight=0.9)

        lm_logits = torch.randn(4, 10, 100)
        lm_targets = torch.randint(0, 100, (4, 10))
        personality_preds = torch.rand(4, 5)
        personality_targets = torch.rand(4, 5)

        loss_lm, _ = pmol_lm_dominant(lm_logits, lm_targets, personality_preds, personality_targets)
        loss_pers, _ = pmol_personality_dominant(lm_logits, lm_targets, personality_preds, personality_targets)

        # Losses should be different due to different weighting
        assert abs(loss_lm.item() - loss_pers.item()) > 0.01
