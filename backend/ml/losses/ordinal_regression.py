"""
Ordinal Regression Loss for Personality Traits

Implements ordinal regression that respects the ordinal nature of personality scores.
Unlike standard regression, this penalizes distant errors more heavily than close errors.

For example:
- Predicting 0.9 when true is 0.2 (error of 0.7) should be penalized MORE than
- Predicting 0.5 when true is 0.2 (error of 0.3)

This is based on the insight that personality traits are continuous but ordinal -
being "very high" vs "very low" is qualitatively different from being "high" vs "medium".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for personality trait prediction.

    This loss function treats personality scores as ordinal values and
    penalizes larger errors more heavily than smaller errors.
    """

    def __init__(
        self,
        num_thresholds: int = 10,
        temperature: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize ordinal regression loss.

        Args:
            num_thresholds: Number of threshold boundaries (10 creates 11 bins for [0.0, 1.0])
            temperature: Temperature for softening the predictions
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_thresholds = num_thresholds
        self.temperature = temperature
        self.reduction = reduction

        # Create threshold boundaries for [0.0, 1.0]
        # Example: for num_thresholds=10, we get [0.1, 0.2, ..., 0.9]
        self.register_buffer(
            'thresholds',
            torch.linspace(0, 1, num_thresholds + 2)[1:-1]
        )

    def _score_to_ordinal_labels(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous scores [0.0, 1.0] to ordinal labels.

        Args:
            scores: Continuous scores [batch_size, num_traits]

        Returns:
            Ordinal labels [batch_size, num_traits, num_thresholds]
            where each element is 1 if score > threshold, else 0
        """
        # scores: [batch_size, num_traits]
        # thresholds: [num_thresholds]
        # Output: [batch_size, num_traits, num_thresholds]

        scores = scores.unsqueeze(-1)  # [batch_size, num_traits, 1]
        thresholds = self.thresholds.unsqueeze(0).unsqueeze(0)  # [1, 1, num_thresholds]

        # Binary labels: 1 if score > threshold, 0 otherwise
        ordinal_labels = (scores > thresholds).float()

        return ordinal_labels

    def _predictions_to_ordinal(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous predictions to ordinal probability distributions.

        Args:
            predictions: Continuous predictions [batch_size, num_traits]

        Returns:
            Ordinal probabilities [batch_size, num_traits, num_thresholds]
        """
        # Similar conversion for predictions
        predictions = predictions.unsqueeze(-1)
        thresholds = self.thresholds.unsqueeze(0).unsqueeze(0)

        # Use sigmoid to get probabilities of exceeding each threshold
        ordinal_probs = torch.sigmoid((predictions - thresholds) / self.temperature)

        return ordinal_probs

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trait_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ordinal regression loss.

        Args:
            predictions: Predicted trait scores [batch_size, num_traits]
            targets: Target trait scores [batch_size, num_traits]
            trait_weights: Optional weights for each trait [num_traits]

        Returns:
            Loss scalar
        """
        # Convert to ordinal representations
        ordinal_targets = self._score_to_ordinal_labels(targets)  # [B, T, N]
        ordinal_preds = self._predictions_to_ordinal(predictions)  # [B, T, N]

        # Binary cross-entropy for each threshold
        # This penalizes incorrect ordinal comparisons
        loss = F.binary_cross_entropy(
            ordinal_preds,
            ordinal_targets,
            reduction='none'
        )  # [B, T, N]

        # Average over thresholds
        loss = loss.mean(dim=-1)  # [B, T]

        # Apply trait weights if provided
        if trait_weights is not None:
            loss = loss * trait_weights.unsqueeze(0)

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedOrdinalMSELoss(nn.Module):
    """
    Combined loss: Ordinal Regression + MSE

    Uses both ordinal regression (to respect ordinality) and MSE (to minimize
    absolute error). The combination provides both ordinal awareness and
    precise value prediction.
    """

    def __init__(
        self,
        ordinal_weight: float = 0.7,
        mse_weight: float = 0.3,
        num_thresholds: int = 10
    ):
        """
        Initialize combined loss.

        Args:
            ordinal_weight: Weight for ordinal regression loss
            mse_weight: Weight for MSE loss
            num_thresholds: Number of thresholds for ordinal loss
        """
        super().__init__()
        self.ordinal_weight = ordinal_weight
        self.mse_weight = mse_weight

        self.ordinal_loss = OrdinalRegressionLoss(
            num_thresholds=num_thresholds,
            reduction='mean'
        )
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            predictions: Predicted scores [batch_size, num_traits]
            targets: Target scores [batch_size, num_traits]

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        ordinal_loss_val = self.ordinal_loss(predictions, targets)
        mse_loss_val = self.mse_loss(predictions, targets)

        total_loss = (
            self.ordinal_weight * ordinal_loss_val +
            self.mse_weight * mse_loss_val
        )

        loss_dict = {
            'ordinal_loss': ordinal_loss_val.item(),
            'mse_loss': mse_loss_val.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


# Testing
def test_ordinal_loss():
    """Test the ordinal regression loss"""
    print("Testing Ordinal Regression Loss")
    print("=" * 50)

    loss_fn = OrdinalRegressionLoss(num_thresholds=10)

    # Test case 1: Perfect prediction
    pred = torch.tensor([[0.5, 0.7, 0.3]])
    target = torch.tensor([[0.5, 0.7, 0.3]])
    loss1 = loss_fn(pred, target)
    print(f"Perfect prediction loss: {loss1.item():.6f}")

    # Test case 2: Small error
    pred = torch.tensor([[0.5, 0.7, 0.3]])
    target = torch.tensor([[0.55, 0.75, 0.35]])
    loss2 = loss_fn(pred, target)
    print(f"Small error loss: {loss2.item():.6f}")

    # Test case 3: Large error (should be much higher)
    pred = torch.tensor([[0.2, 0.3, 0.2]])
    target = torch.tensor([[0.9, 0.9, 0.8]])
    loss3 = loss_fn(pred, target)
    print(f"Large error loss: {loss3.item():.6f}")

    print("\nExpected: loss1 < loss2 << loss3")
    print(f"Verified: {loss1 < loss2 < loss3}")

    # Test combined loss
    print("\n" + "=" * 50)
    print("Testing Combined Ordinal + MSE Loss")
    print("=" * 50)

    combined_loss_fn = CombinedOrdinalMSELoss()

    pred = torch.tensor([[0.5, 0.7, 0.3]])
    target = torch.tensor([[0.8, 0.9, 0.6]])
    loss, loss_dict = combined_loss_fn(pred, target)

    print(f"Total loss: {loss.item():.6f}")
    print(f"Loss breakdown: {loss_dict}")


if __name__ == "__main__":
    test_ordinal_loss()
