"""
Mixture of Losses (MoL) Implementation

Implements the Mixture of Losses approach from "Architecting the Artificial Psychologist":
1. Cross-Entropy Loss for domain-specific data (car sales)
2. KL Divergence Loss for general personality data
3. Dynamic weighting based on data source

This allows the model to:
- Learn domain-specific patterns from car data (via CE)
- Maintain general personality knowledge from public data (via KL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from enum import Enum


class DataSourceType(Enum):
    """Types of data sources for MoL"""
    PUBLIC_GENERAL = "public_general"
    CAR_DOMAIN = "car_domain"
    SYNTHETIC = "synthetic"


class MixtureOfLosses(nn.Module):
    """
    Mixture of Losses (MoL) implementation.

    Applies different loss functions based on the data source:
    - General data: KL Divergence (preserve general personality knowledge)
    - Domain data: Cross-Entropy (learn specific patterns)
    - Synthetic data: Combined (learn from teacher model)
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        kl_weight: float = 0.5,
        temperature: float = 1.0,
        label_smoothing: float = 0.1
    ):
        """
        Initialize MoL.

        Args:
            ce_weight: Weight for cross-entropy loss (domain data)
            kl_weight: Weight for KL divergence loss (general data)
            temperature: Temperature for KL divergence
            label_smoothing: Label smoothing factor for CE loss
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.kl_weight = kl_weight
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        source_types: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute mixture of losses.

        Args:
            logits: Model predictions [batch_size, vocab_size]
            targets: Target tokens [batch_size]
            source_types: Data source type for each sample [batch_size]
                         (0=public_general, 1=car_domain, 2=synthetic)
            teacher_logits: Optional teacher model logits for KL divergence

        Returns:
            Tuple of (total_loss, loss_dict with breakdown)
        """
        batch_size = logits.size(0)

        # Separate samples by source type
        public_mask = (source_types == 0)
        car_mask = (source_types == 1)
        synthetic_mask = (source_types == 2)

        total_loss = 0.0
        loss_dict = {
            'ce_loss': 0.0,
            'kl_loss': 0.0,
            'num_public': public_mask.sum().item(),
            'num_car': car_mask.sum().item(),
            'num_synthetic': synthetic_mask.sum().item()
        }

        # 1. Cross-Entropy for car domain data
        if car_mask.any():
            car_logits = logits[car_mask]
            car_targets = targets[car_mask]

            ce_loss = self.ce_loss(car_logits, car_targets)
            total_loss = total_loss + self.ce_weight * ce_loss
            loss_dict['ce_loss'] = ce_loss.item()

        # 2. KL Divergence for public general data
        if public_mask.any():
            public_logits = logits[public_mask]

            if teacher_logits is not None:
                # Use teacher model as reference distribution
                public_teacher_logits = teacher_logits[public_mask]

                # Apply temperature scaling
                log_probs = F.log_softmax(public_logits / self.temperature, dim=-1)
                teacher_probs = F.softmax(public_teacher_logits / self.temperature, dim=-1)

                kl_loss = self.kl_loss(log_probs, teacher_probs)
            else:
                # Fallback: use CE loss if no teacher logits
                public_targets = targets[public_mask]
                kl_loss = self.ce_loss(public_logits, public_targets)

            total_loss = total_loss + self.kl_weight * kl_loss
            loss_dict['kl_loss'] = kl_loss.item()

        # 3. Combined loss for synthetic data
        if synthetic_mask.any():
            synthetic_logits = logits[synthetic_mask]
            synthetic_targets = targets[synthetic_mask]

            # Use both CE and KL for synthetic data
            synth_ce = self.ce_loss(synthetic_logits, synthetic_targets)

            if teacher_logits is not None:
                synthetic_teacher_logits = teacher_logits[synthetic_mask]
                log_probs = F.log_softmax(synthetic_logits / self.temperature, dim=-1)
                teacher_probs = F.softmax(synthetic_teacher_logits / self.temperature, dim=-1)
                synth_kl = self.kl_loss(log_probs, teacher_probs)
            else:
                synth_kl = 0.0

            synth_loss = 0.7 * synth_ce + 0.3 * (synth_kl if isinstance(synth_kl, torch.Tensor) else torch.tensor(0.0))
            total_loss = total_loss + synth_loss
            loss_dict['synthetic_loss'] = synth_loss.item() if isinstance(synth_loss, torch.Tensor) else synth_loss

        # Normalize by batch size
        if batch_size > 0:
            total_loss = total_loss / batch_size

        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        return total_loss, loss_dict


class PersonalityMoL(nn.Module):
    """
    Specialized MoL for personality trait prediction.

    Combines:
    1. Cross-Entropy for next-token prediction (language modeling)
    2. Ordinal Regression for personality scores
    3. Source-aware weighting
    """

    def __init__(
        self,
        lm_weight: float = 0.6,
        personality_weight: float = 0.4,
        ordinal_loss: Optional[nn.Module] = None
    ):
        """
        Initialize Personality MoL.

        Args:
            lm_weight: Weight for language modeling loss
            personality_weight: Weight for personality prediction loss
            ordinal_loss: Optional ordinal regression loss module
        """
        super().__init__()
        self.lm_weight = lm_weight
        self.personality_weight = personality_weight

        self.lm_loss = nn.CrossEntropyLoss()

        # Import here to avoid circular dependency
        if ordinal_loss is None:
            from .ordinal_regression import CombinedOrdinalMSELoss
            self.personality_loss = CombinedOrdinalMSELoss()
        else:
            self.personality_loss = ordinal_loss

    def forward(
        self,
        lm_logits: torch.Tensor,
        lm_targets: torch.Tensor,
        personality_preds: Optional[torch.Tensor] = None,
        personality_targets: Optional[torch.Tensor] = None,
        source_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss for language modeling and personality prediction.

        Args:
            lm_logits: Language model logits [batch_size, seq_len, vocab_size]
            lm_targets: Target tokens [batch_size, seq_len]
            personality_preds: Predicted personality scores [batch_size, 5]
            personality_targets: Target personality scores [batch_size, 5]
            source_types: Data source types [batch_size]

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # 1. Language modeling loss
        if lm_logits.dim() == 3:
            # Flatten for cross-entropy
            lm_logits_flat = lm_logits.view(-1, lm_logits.size(-1))
            lm_targets_flat = lm_targets.view(-1)
        else:
            lm_logits_flat = lm_logits
            lm_targets_flat = lm_targets

        lm_loss_val = self.lm_loss(lm_logits_flat, lm_targets_flat)

        total_loss = self.lm_weight * lm_loss_val

        loss_dict = {
            'lm_loss': lm_loss_val.item(),
            'total_loss': 0.0
        }

        # 2. Personality prediction loss (if available)
        if personality_preds is not None and personality_targets is not None:
            personality_loss_val, personality_loss_dict = self.personality_loss(
                personality_preds,
                personality_targets
            )

            total_loss = total_loss + self.personality_weight * personality_loss_val

            loss_dict.update({
                f'personality_{k}': v for k, v in personality_loss_dict.items()
            })

        # 3. Apply source-specific weighting if provided
        if source_types is not None:
            # Weight car domain samples higher
            car_mask = (source_types == 1)
            if car_mask.any():
                car_boost = 1.2  # 20% boost for car domain
                total_loss = total_loss * (1 + (car_boost - 1) * car_mask.float().mean())

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


def test_mol():
    """Test Mixture of Losses"""
    print("Testing Mixture of Losses (MoL)")
    print("=" * 50)

    batch_size = 4
    vocab_size = 1000

    # Create dummy data
    logits = torch.randn(batch_size, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size,))
    source_types = torch.tensor([0, 1, 0, 2])  # public, car, public, synthetic

    mol = MixtureOfLosses()

    # Without teacher logits
    loss, loss_dict = mol(logits, targets, source_types)
    print(f"\nLoss without teacher: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")

    # With teacher logits
    teacher_logits = torch.randn(batch_size, vocab_size)
    loss, loss_dict = mol(logits, targets, source_types, teacher_logits)
    print(f"\nLoss with teacher: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")

    # Test Personality MoL
    print("\n" + "=" * 50)
    print("Testing Personality MoL")
    print("=" * 50)

    seq_len = 10
    lm_logits = torch.randn(batch_size, seq_len, vocab_size)
    lm_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    personality_preds = torch.rand(batch_size, 5)
    personality_targets = torch.rand(batch_size, 5)

    personality_mol = PersonalityMoL()

    loss, loss_dict = personality_mol(
        lm_logits,
        lm_targets,
        personality_preds,
        personality_targets
    )

    print(f"\nPersonality MoL loss: {loss.item():.4f}")
    print(f"Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    test_mol()
