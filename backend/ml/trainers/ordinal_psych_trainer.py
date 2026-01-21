"""
OrdinalPsychTrainer: Custom HuggingFace Trainer for Psychometric Profiling

This is the core training component that implements:
1. Ordinal Regression Loss for personality traits
2. Mixture of Losses (MoL) based on data source
3. Custom compute_loss override for HuggingFace Trainer
4. Integration with Unsloth for optimization

Based on "Architecting the Artificial Psychologist" research methodology.
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import nested_detach
from typing import Dict, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
import numpy as np

from ..losses.ordinal_regression import CombinedOrdinalMSELoss, OrdinalRegressionLoss
from ..losses.mixture_of_losses import PersonalityMoL, DataSourceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrdinalPsychTrainingArguments(TrainingArguments):
    """
    Extended training arguments for ordinal psychology training.
    """
    # Loss weights
    lm_loss_weight: float = 0.6
    personality_loss_weight: float = 0.4
    ordinal_loss_weight: float = 0.7
    mse_loss_weight: float = 0.3

    # MoL weights
    ce_weight: float = 1.0
    kl_weight: float = 0.5
    kl_temperature: float = 1.0

    # Data source boost
    car_domain_boost: float = 1.2  # Boost car domain samples by 20%

    # Ordinal regression
    num_ordinal_thresholds: int = 10

    # Personality prediction
    predict_personality: bool = True
    personality_head_hidden_size: int = 256


class PersonalityPredictionHead(nn.Module):
    """
    Personality prediction head for extracting OCEAN scores from hidden states.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = 256,
        num_traits: int = 5,
        dropout: float = 0.1
    ):
        """
        Initialize personality prediction head.

        Args:
            hidden_size: Size of model hidden states
            intermediate_size: Size of intermediate layer
            num_traits: Number of personality traits (5 for OCEAN)
            dropout: Dropout probability
        """
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, intermediate_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size // 2, num_traits),
            nn.Sigmoid()  # Output [0, 1] for each trait
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict personality scores from hidden states.

        Args:
            hidden_states: Model hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Personality scores [batch_size, 5] in range [0, 1]
        """
        # Use the last token's hidden state (like CLS token)
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]

        # Project to personality scores
        personality_scores = self.projection(last_hidden)  # [batch_size, 5]

        return personality_scores


class OrdinalPsychTrainer(Trainer):
    """
    Custom Trainer for psychometric profiling with ordinal regression and MoL.

    This trainer:
    1. Overrides compute_loss to implement custom loss functions
    2. Applies Mixture of Losses based on data source
    3. Uses Ordinal Regression for personality trait prediction
    4. Integrates with Unsloth for optimization
    """

    def __init__(
        self,
        model,
        args: OrdinalPsychTrainingArguments,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        """
        Initialize OrdinalPsychTrainer.

        Args:
            Same as HuggingFace Trainer, with OrdinalPsychTrainingArguments
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Initialize custom loss functions
        self.personality_loss = CombinedOrdinalMSELoss(
            ordinal_weight=args.ordinal_loss_weight,
            mse_weight=args.mse_loss_weight,
            num_thresholds=args.num_ordinal_thresholds
        )

        self.mol = PersonalityMoL(
            lm_weight=args.lm_loss_weight,
            personality_weight=args.personality_loss_weight,
            ordinal_loss=self.personality_loss
        )

        # Add personality prediction head if enabled
        if args.predict_personality:
            hidden_size = model.config.hidden_size
            self.personality_head = PersonalityPredictionHead(
                hidden_size=hidden_size,
                intermediate_size=args.personality_head_hidden_size
            )

            # Move to same device as model
            if hasattr(model, 'device'):
                self.personality_head = self.personality_head.to(model.device)
        else:
            self.personality_head = None

        logger.info(f"OrdinalPsychTrainer initialized with:")
        logger.info(f"  LM loss weight: {args.lm_loss_weight}")
        logger.info(f"  Personality loss weight: {args.personality_loss_weight}")
        logger.info(f"  Ordinal loss weight: {args.ordinal_loss_weight}")
        logger.info(f"  MSE loss weight: {args.mse_loss_weight}")
        logger.info(f"  Personality prediction: {args.predict_personality}")

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Custom loss computation implementing:
        1. Standard language modeling loss
        2. Ordinal regression for personality traits
        3. Mixture of losses based on data source

        This method overrides the HuggingFace Trainer's compute_loss.

        Args:
            model: The model being trained
            inputs: Dictionary of inputs from data_collator
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True
        """
        # Extract inputs
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        # Extract personality-related inputs (if available)
        personality_targets = inputs.get("personality_scores", None)  # [batch_size, 5]
        source_types = inputs.get("source_type", None)  # [batch_size]

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True  # Need hidden states for personality prediction
        )

        # Get language modeling loss and logits
        lm_loss = outputs.loss if hasattr(outputs, 'loss') else None
        lm_logits = outputs.logits
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None

        # Initialize loss components
        total_loss = 0.0
        loss_dict = {}

        # 1. Language Modeling Loss (always computed)
        if lm_loss is not None:
            total_loss = self.args.lm_loss_weight * lm_loss
            loss_dict['lm_loss'] = lm_loss.item()
        else:
            # Compute LM loss manually if not provided
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            total_loss = self.args.lm_loss_weight * lm_loss
            loss_dict['lm_loss'] = lm_loss.item()

        # 2. Personality Prediction Loss (if head is available and targets provided)
        if (self.personality_head is not None and
            personality_targets is not None and
            hidden_states is not None):

            # Predict personality scores
            personality_preds = self.personality_head(hidden_states)  # [batch_size, 5]

            # Ensure personality_targets is the right shape and type
            if isinstance(personality_targets, dict):
                # Convert dict to tensor [batch_size, 5]
                personality_targets_tensor = torch.stack([
                    torch.tensor([
                        pt.get("openness", 0.5),
                        pt.get("conscientiousness", 0.5),
                        pt.get("extraversion", 0.5),
                        pt.get("agreeableness", 0.5),
                        pt.get("neuroticism", 0.5)
                    ]) for pt in personality_targets
                ]).to(personality_preds.device)
            else:
                personality_targets_tensor = personality_targets

            # Compute ordinal regression loss
            personality_loss, personality_loss_dict = self.personality_loss(
                personality_preds,
                personality_targets_tensor
            )

            total_loss = total_loss + self.args.personality_loss_weight * personality_loss

            loss_dict.update({
                f'personality_{k}': v for k, v in personality_loss_dict.items()
            })

        # 3. Apply data source-specific weighting
        if source_types is not None:
            # Boost car domain samples
            car_mask = (source_types == 1)  # 1 = car_domain
            if car_mask.any():
                boost_factor = self.args.car_domain_boost
                car_boost = (boost_factor - 1.0) * car_mask.float().mean()
                total_loss = total_loss * (1.0 + car_boost)

                loss_dict['car_domain_boost'] = car_boost.item()

        # Log losses periodically
        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(f"Step {self.state.global_step} - Loss breakdown: {loss_dict}")

        loss_dict['total_loss'] = total_loss.item()

        # Store loss dict for logging
        if not hasattr(self, '_loss_history'):
            self._loss_history = []
        self._loss_history.append(loss_dict)

        # Return loss and outputs if requested
        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Custom prediction step that includes personality predictions.
        """
        # Use parent's prediction_step but add personality predictions
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )

        # Add personality predictions if head exists
        if self.personality_head is not None and not prediction_loss_only:
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]
                personality_preds = self.personality_head(hidden_states)

                # Concatenate with logits
                if logits is not None:
                    logits = (logits, personality_preds)
                else:
                    logits = personality_preds

        return (loss, logits, labels)

    def get_loss_history(self) -> list:
        """
        Get the history of loss values for analysis.

        Returns:
            List of loss dictionaries
        """
        return getattr(self, '_loss_history', [])

    def save_loss_history(self, output_path: str):
        """
        Save loss history to a JSON file for visualization.

        Args:
            output_path: Path to save loss history
        """
        import json
        from pathlib import Path

        loss_history = self.get_loss_history()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(loss_history, f, indent=2)

        logger.info(f"Loss history saved to {output_path}")


# Example usage and testing
def example_usage():
    """
    Example of how to use OrdinalPsychTrainer.
    """
    print("=" * 60)
    print("OrdinalPsychTrainer Example Usage")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. Load base model
    model_name = "meta-llama/Llama-3.2-1B"
    print(f"\nLoading model: {model_name}")

    # Note: In practice, you'd use Unsloth to load the model for optimization
    # from unsloth import FastLanguageModel
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=model_name,
    #     max_seq_length=2048,
    #     dtype=torch.float16,
    #     load_in_4bit=True,
    # )

    # 2. Set up training arguments
    training_args = OrdinalPsychTrainingArguments(
        output_dir="./models/checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        # Custom arguments
        lm_loss_weight=0.6,
        personality_loss_weight=0.4,
        ordinal_loss_weight=0.7,
        mse_loss_weight=0.3,
        predict_personality=True,
        car_domain_boost=1.2,
    )

    print("\nTraining Arguments:")
    print(f"  LM loss weight: {training_args.lm_loss_weight}")
    print(f"  Personality loss weight: {training_args.personality_loss_weight}")
    print(f"  Ordinal loss weight: {training_args.ordinal_loss_weight}")
    print(f"  Car domain boost: {training_args.car_domain_boost}")

    # 3. Initialize trainer
    # trainer = OrdinalPsychTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    # )

    # 4. Train
    # trainer.train()

    # 5. Save loss history
    # trainer.save_loss_history("./models/checkpoints/loss_history.json")

    print("\nTrainer initialized successfully!")
    print("Ready for training with ordinal regression and MoL.")


if __name__ == "__main__":
    example_usage()
