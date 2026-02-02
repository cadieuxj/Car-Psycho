"""
Unit Tests for Ordinal Regression Loss Functions

Tests cover:
1. OrdinalRegressionLoss initialization
2. Score to ordinal labels conversion
3. Predictions to ordinal conversion
4. Forward pass computation
5. CombinedOrdinalMSELoss
6. Gradient computation
7. Edge cases
"""

import pytest
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.ml.losses.ordinal_regression import (
    OrdinalRegressionLoss,
    CombinedOrdinalMSELoss
)


class TestOrdinalRegressionLossInitialization:
    """Tests for OrdinalRegressionLoss initialization."""

    def test_default_initialization(self):
        """Test default initialization parameters."""
        loss_fn = OrdinalRegressionLoss()
        assert loss_fn.num_thresholds == 10
        assert loss_fn.temperature == 1.0
        assert loss_fn.reduction == 'mean'

    def test_custom_num_thresholds(self):
        """Test custom number of thresholds."""
        loss_fn = OrdinalRegressionLoss(num_thresholds=5)
        assert loss_fn.num_thresholds == 5
        assert len(loss_fn.thresholds) == 5

    def test_custom_temperature(self):
        """Test custom temperature."""
        loss_fn = OrdinalRegressionLoss(temperature=2.0)
        assert loss_fn.temperature == 2.0

    def test_custom_reduction(self):
        """Test custom reduction modes."""
        for reduction in ['mean', 'sum', 'none']:
            loss_fn = OrdinalRegressionLoss(reduction=reduction)
            assert loss_fn.reduction == reduction

    def test_thresholds_are_evenly_spaced(self):
        """Test that thresholds are evenly spaced."""
        loss_fn = OrdinalRegressionLoss(num_thresholds=10)
        thresholds = loss_fn.thresholds

        # Check that thresholds are evenly spaced
        diffs = thresholds[1:] - thresholds[:-1]
        assert torch.allclose(diffs, diffs[0] * torch.ones_like(diffs), atol=1e-6)

    def test_thresholds_in_valid_range(self):
        """Test that thresholds are in (0, 1) range."""
        loss_fn = OrdinalRegressionLoss(num_thresholds=10)
        thresholds = loss_fn.thresholds

        assert (thresholds > 0).all()
        assert (thresholds < 1).all()


class TestScoreToOrdinalLabels:
    """Tests for _score_to_ordinal_labels method."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function for tests."""
        return OrdinalRegressionLoss(num_thresholds=10)

    def test_low_score_ordinal_labels(self, loss_fn):
        """Test ordinal labels for low scores."""
        scores = torch.tensor([[0.05]])  # Very low score
        labels = loss_fn._score_to_ordinal_labels(scores)

        # Should be mostly zeros (below most thresholds)
        assert labels.shape == (1, 1, 10)
        assert labels.sum() < 2  # At most 1 threshold exceeded

    def test_high_score_ordinal_labels(self, loss_fn):
        """Test ordinal labels for high scores."""
        scores = torch.tensor([[0.95]])  # Very high score
        labels = loss_fn._score_to_ordinal_labels(scores)

        # Should be mostly ones (above most thresholds)
        assert labels.sum() > 8  # At least 9 thresholds exceeded

    def test_mid_score_ordinal_labels(self, loss_fn):
        """Test ordinal labels for mid-range scores."""
        scores = torch.tensor([[0.5]])
        labels = loss_fn._score_to_ordinal_labels(scores)

        # Should be about half ones and half zeros
        assert 4 <= labels.sum() <= 6

    def test_batch_processing(self, loss_fn):
        """Test batch processing of ordinal labels."""
        batch_size = 4
        num_traits = 5
        scores = torch.rand(batch_size, num_traits)
        labels = loss_fn._score_to_ordinal_labels(scores)

        assert labels.shape == (batch_size, num_traits, 10)

    def test_ordinal_labels_are_binary(self, loss_fn):
        """Test that ordinal labels are binary (0 or 1)."""
        scores = torch.rand(4, 5)
        labels = loss_fn._score_to_ordinal_labels(scores)

        assert ((labels == 0) | (labels == 1)).all()

    def test_ordinal_labels_are_monotonic(self, loss_fn):
        """Test that ordinal labels are monotonically decreasing."""
        scores = torch.tensor([[0.5]])
        labels = loss_fn._score_to_ordinal_labels(scores)

        # For a single score, labels should be 1,1,...,0,0 (monotonic)
        # Each subsequent threshold should have <= previous label value
        for i in range(labels.shape[-1] - 1):
            assert (labels[..., i] >= labels[..., i + 1]).all()


class TestPredictionsToOrdinal:
    """Tests for _predictions_to_ordinal method."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function for tests."""
        return OrdinalRegressionLoss(num_thresholds=10)

    def test_low_prediction_ordinal(self, loss_fn):
        """Test ordinal probabilities for low predictions."""
        preds = torch.tensor([[0.1]])
        ordinal_probs = loss_fn._predictions_to_ordinal(preds)

        # Should have low probabilities of exceeding thresholds
        assert ordinal_probs.mean() < 0.3

    def test_high_prediction_ordinal(self, loss_fn):
        """Test ordinal probabilities for high predictions."""
        preds = torch.tensor([[0.9]])
        ordinal_probs = loss_fn._predictions_to_ordinal(preds)

        # Should have high probabilities of exceeding thresholds
        assert ordinal_probs.mean() > 0.7

    def test_ordinal_probs_in_valid_range(self, loss_fn):
        """Test that ordinal probabilities are in [0, 1]."""
        preds = torch.rand(4, 5)
        ordinal_probs = loss_fn._predictions_to_ordinal(preds)

        assert (ordinal_probs >= 0).all()
        assert (ordinal_probs <= 1).all()

    def test_temperature_effect(self):
        """Test that temperature affects ordinal probabilities."""
        preds = torch.tensor([[0.5]])

        loss_fn_temp1 = OrdinalRegressionLoss(temperature=1.0)
        loss_fn_temp01 = OrdinalRegressionLoss(temperature=0.1)

        probs_temp1 = loss_fn_temp1._predictions_to_ordinal(preds)
        probs_temp01 = loss_fn_temp01._predictions_to_ordinal(preds)

        # Lower temperature should produce sharper (more extreme) probabilities
        # Higher temperature should produce smoother probabilities
        assert probs_temp01.std() > probs_temp1.std()


class TestOrdinalRegressionLossForward:
    """Tests for OrdinalRegressionLoss forward pass."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function for tests."""
        return OrdinalRegressionLoss(num_thresholds=10)

    def test_perfect_prediction_low_loss(self, loss_fn):
        """Test that perfect prediction produces low loss."""
        targets = torch.tensor([[0.5, 0.7, 0.3]])
        predictions = torch.tensor([[0.5, 0.7, 0.3]])

        loss = loss_fn(predictions, targets)
        assert loss < 0.1  # Should be very low

    def test_bad_prediction_high_loss(self, loss_fn):
        """Test that bad prediction produces high loss."""
        targets = torch.tensor([[0.9, 0.9, 0.9]])
        predictions = torch.tensor([[0.1, 0.1, 0.1]])

        loss = loss_fn(predictions, targets)
        assert loss > 0.5  # Should be high

    def test_loss_ordering(self, loss_fn):
        """Test that larger errors produce larger losses."""
        target = torch.tensor([[0.5]])

        pred_perfect = torch.tensor([[0.5]])
        pred_small_error = torch.tensor([[0.6]])
        pred_large_error = torch.tensor([[0.9]])

        loss_perfect = loss_fn(pred_perfect, target)
        loss_small = loss_fn(pred_small_error, target)
        loss_large = loss_fn(pred_large_error, target)

        assert loss_perfect < loss_small < loss_large

    def test_loss_reduction_mean(self):
        """Test mean reduction."""
        loss_fn = OrdinalRegressionLoss(reduction='mean')
        predictions = torch.rand(4, 5)
        targets = torch.rand(4, 5)

        loss = loss_fn(predictions, targets)
        assert loss.dim() == 0  # Scalar

    def test_loss_reduction_sum(self):
        """Test sum reduction."""
        loss_fn = OrdinalRegressionLoss(reduction='sum')
        predictions = torch.rand(4, 5)
        targets = torch.rand(4, 5)

        loss = loss_fn(predictions, targets)
        assert loss.dim() == 0  # Scalar

    def test_loss_reduction_none(self):
        """Test no reduction."""
        loss_fn = OrdinalRegressionLoss(reduction='none')
        predictions = torch.rand(4, 5)
        targets = torch.rand(4, 5)

        loss = loss_fn(predictions, targets)
        assert loss.shape == (4, 5)  # Per-sample, per-trait losses

    def test_trait_weights(self, loss_fn):
        """Test trait weights application."""
        predictions = torch.rand(4, 5)
        targets = torch.rand(4, 5)
        trait_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0])

        loss_unweighted = loss_fn(predictions, targets)
        loss_weighted = loss_fn(predictions, targets, trait_weights)

        # Weighted loss should be different (unless all weights are equal)
        assert loss_weighted != loss_unweighted

    def test_loss_is_differentiable(self, loss_fn):
        """Test that loss is differentiable."""
        predictions = torch.rand(4, 5, requires_grad=True)
        targets = torch.rand(4, 5)

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.isnan(predictions.grad).any()


class TestCombinedOrdinalMSELoss:
    """Tests for CombinedOrdinalMSELoss."""

    def test_default_initialization(self):
        """Test default initialization."""
        loss_fn = CombinedOrdinalMSELoss()
        assert loss_fn.ordinal_weight == 0.7
        assert loss_fn.mse_weight == 0.3

    def test_custom_weights(self):
        """Test custom weight initialization."""
        loss_fn = CombinedOrdinalMSELoss(ordinal_weight=0.5, mse_weight=0.5)
        assert loss_fn.ordinal_weight == 0.5
        assert loss_fn.mse_weight == 0.5

    def test_forward_returns_tuple(self):
        """Test that forward returns loss and dict."""
        loss_fn = CombinedOrdinalMSELoss()
        predictions = torch.rand(4, 5)
        targets = torch.rand(4, 5)

        result = loss_fn(predictions, targets)
        assert isinstance(result, tuple)
        assert len(result) == 2

        total_loss, loss_dict = result
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_dict, dict)

    def test_loss_dict_contains_components(self):
        """Test that loss dict contains component losses."""
        loss_fn = CombinedOrdinalMSELoss()
        predictions = torch.rand(4, 5)
        targets = torch.rand(4, 5)

        _, loss_dict = loss_fn(predictions, targets)

        assert 'ordinal_loss' in loss_dict
        assert 'mse_loss' in loss_dict
        assert 'total_loss' in loss_dict

    def test_total_loss_is_weighted_sum(self):
        """Test that total loss is weighted sum of components."""
        loss_fn = CombinedOrdinalMSELoss(ordinal_weight=0.6, mse_weight=0.4)
        predictions = torch.rand(4, 5)
        targets = torch.rand(4, 5)

        total_loss, loss_dict = loss_fn(predictions, targets)

        expected_total = 0.6 * loss_dict['ordinal_loss'] + 0.4 * loss_dict['mse_loss']
        assert abs(total_loss.item() - expected_total) < 1e-5

    def test_combined_loss_perfect_prediction(self):
        """Test combined loss with perfect prediction."""
        loss_fn = CombinedOrdinalMSELoss()
        targets = torch.tensor([[0.5, 0.7, 0.3, 0.8, 0.2]])
        predictions = torch.tensor([[0.5, 0.7, 0.3, 0.8, 0.2]])

        total_loss, loss_dict = loss_fn(predictions, targets)

        assert loss_dict['mse_loss'] < 1e-6  # MSE should be ~0
        assert total_loss.item() < 0.1

    def test_combined_loss_is_differentiable(self):
        """Test that combined loss is differentiable."""
        loss_fn = CombinedOrdinalMSELoss()
        predictions = torch.rand(4, 5, requires_grad=True)
        targets = torch.rand(4, 5)

        total_loss, _ = loss_fn(predictions, targets)
        total_loss.backward()

        assert predictions.grad is not None
        assert not torch.isnan(predictions.grad).any()


class TestOrdinalLossEdgeCases:
    """Tests for edge cases in ordinal loss functions."""

    def test_single_sample(self):
        """Test with single sample."""
        loss_fn = OrdinalRegressionLoss()
        predictions = torch.tensor([[0.5]])
        targets = torch.tensor([[0.5]])

        loss = loss_fn(predictions, targets)
        assert not torch.isnan(loss)
        assert loss.dim() == 0

    def test_single_trait(self):
        """Test with single trait per sample."""
        loss_fn = OrdinalRegressionLoss()
        predictions = torch.rand(10, 1)
        targets = torch.rand(10, 1)

        loss = loss_fn(predictions, targets)
        assert not torch.isnan(loss)

    def test_boundary_values(self):
        """Test with boundary values (0 and 1)."""
        loss_fn = OrdinalRegressionLoss()
        predictions = torch.tensor([[0.0, 1.0]])
        targets = torch.tensor([[0.0, 1.0]])

        loss = loss_fn(predictions, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_large_batch(self):
        """Test with large batch size."""
        loss_fn = OrdinalRegressionLoss()
        predictions = torch.rand(1000, 5)
        targets = torch.rand(1000, 5)

        loss = loss_fn(predictions, targets)
        assert not torch.isnan(loss)

    def test_different_thresholds(self):
        """Test with different number of thresholds."""
        for num_thresholds in [1, 5, 10, 20, 50]:
            loss_fn = OrdinalRegressionLoss(num_thresholds=num_thresholds)
            predictions = torch.rand(4, 5)
            targets = torch.rand(4, 5)

            loss = loss_fn(predictions, targets)
            assert not torch.isnan(loss)

    def test_gpu_compatibility(self):
        """Test GPU compatibility if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        loss_fn = OrdinalRegressionLoss().cuda()
        predictions = torch.rand(4, 5).cuda()
        targets = torch.rand(4, 5).cuda()

        loss = loss_fn(predictions, targets)
        assert loss.device.type == 'cuda'
        assert not torch.isnan(loss)


class TestOrdinalLossSymmetry:
    """Tests for loss function properties."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function for tests."""
        return OrdinalRegressionLoss()

    def test_symmetric_errors(self, loss_fn):
        """Test that symmetric errors produce similar losses."""
        target = torch.tensor([[0.5]])

        pred_above = torch.tensor([[0.7]])  # +0.2 error
        pred_below = torch.tensor([[0.3]])  # -0.2 error

        loss_above = loss_fn(pred_above, target)
        loss_below = loss_fn(pred_below, target)

        # Losses should be similar (not necessarily equal due to threshold positions)
        assert abs(loss_above.item() - loss_below.item()) < 0.1

    def test_ordinal_property(self, loss_fn):
        """Test ordinal property: penalize distant errors more."""
        target = torch.tensor([[0.2]])

        pred_close = torch.tensor([[0.4]])  # Close error
        pred_far = torch.tensor([[0.9]])  # Far error

        loss_close = loss_fn(pred_close, target)
        loss_far = loss_fn(pred_far, target)

        # Far prediction should have much higher loss
        assert loss_far > loss_close * 1.5
