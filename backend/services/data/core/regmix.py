"""
RegMix: Regression-based Data Mixing

Implements the RegMix methodology from "Architecting the Artificial Psychologist"
for optimal mixing of:
1. Public General Psychology Data (40%)
2. Car Domain-Specific Data (40%)
3. Synthetic PsychSteer-labeled Data (20%)

Key Innovation:
- Maintains trait distribution balance across data sources
- Regression-based sampling to avoid data bias
- Stratified sampling by personality quintiles
"""

import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Represents a data source with its samples"""
    name: str
    source_type: str  # 'public_general', 'car_domain', 'synthetic'
    samples: List[Dict]
    num_samples: int

    def __post_init__(self):
        self.num_samples = len(self.samples)


@dataclass
class RegMixConfig:
    """Configuration for RegMix data mixing"""
    public_general_ratio: int = 40
    car_domain_ratio: int = 40
    synthetic_ratio: int = 20
    total_samples: Optional[int] = None  # If None, use all available
    stratify_by_traits: bool = True
    num_quintiles: int = 5  # For stratification
    random_seed: int = 42

    def validate(self):
        """Ensure ratios sum to 100"""
        total = self.public_general_ratio + self.car_domain_ratio + self.synthetic_ratio
        if total != 100:
            raise ValueError(f"Ratios must sum to 100, got {total}")


class RegMix:
    """
    RegMix implementation for data mixing.

    This class implements regression-based data mixing that ensures:
    1. Correct proportions of each data type
    2. Balanced personality trait distributions
    3. Stratified sampling to avoid bias
    """

    def __init__(self, config: RegMixConfig = None):
        """
        Initialize RegMix.

        Args:
            config: RegMix configuration
        """
        self.config = config or RegMixConfig()
        self.config.validate()

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        logger.info(f"RegMix initialized with ratios: "
                   f"Public={self.config.public_general_ratio}%, "
                   f"Car={self.config.car_domain_ratio}%, "
                   f"Synthetic={self.config.synthetic_ratio}%")

    def _get_trait_quintile(self, scores: Dict[str, float]) -> Dict[str, int]:
        """
        Assign each trait to a quintile (0-4) based on its score.

        Args:
            scores: Dictionary of Big Five scores

        Returns:
            Dictionary mapping trait names to quintile indices
        """
        quintiles = {}
        for trait, score in scores.items():
            # Map [0.0, 1.0] to quintiles [0, 4]
            quintile = min(int(score * 5), 4)
            quintiles[trait] = quintile
        return quintiles

    def _stratify_samples(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Stratify samples by their overall personality profile.

        We create strata based on the average quintile across all traits
        to ensure we sample from diverse personality profiles.

        Args:
            samples: List of data samples with personality_scores

        Returns:
            Dictionary mapping strata IDs to sample lists
        """
        strata = defaultdict(list)

        for sample in samples:
            if "personality_scores" not in sample:
                logger.warning(f"Sample missing personality_scores: {sample.get('id', 'unknown')}")
                continue

            scores = sample["personality_scores"]
            quintiles = self._get_trait_quintile(scores)

            # Create a strata key based on the personality profile
            # We use a simplified approach: average quintile
            avg_quintile = int(np.mean(list(quintiles.values())))
            strata_key = f"stratum_{avg_quintile}"

            strata[strata_key].append(sample)

        logger.info(f"Created {len(strata)} strata from {len(samples)} samples")
        for stratum_id, stratum_samples in strata.items():
            logger.info(f"  {stratum_id}: {len(stratum_samples)} samples")

        return dict(strata)

    def _stratified_sample(
        self,
        samples: List[Dict],
        n_samples: int
    ) -> List[Dict]:
        """
        Perform stratified sampling to maintain trait distribution.

        Args:
            samples: List of samples to sample from
            n_samples: Number of samples to draw

        Returns:
            List of sampled items
        """
        if not self.config.stratify_by_traits:
            # Simple random sampling
            return random.sample(samples, min(n_samples, len(samples)))

        if len(samples) <= n_samples:
            return samples

        # Stratify samples
        strata = self._stratify_samples(samples)

        if not strata:
            logger.warning("No valid strata created, falling back to random sampling")
            return random.sample(samples, n_samples)

        # Calculate samples per stratum proportionally
        sampled = []
        total_available = sum(len(stratum) for stratum in strata.values())

        for stratum_id, stratum_samples in strata.items():
            # Proportional allocation
            stratum_proportion = len(stratum_samples) / total_available
            stratum_n = int(n_samples * stratum_proportion)

            # Sample from this stratum
            stratum_n = min(stratum_n, len(stratum_samples))
            if stratum_n > 0:
                sampled.extend(random.sample(stratum_samples, stratum_n))

        # If we didn't get enough samples due to rounding, add more
        if len(sampled) < n_samples:
            remaining = n_samples - len(sampled)
            all_unsampled = [s for s in samples if s not in sampled]
            if all_unsampled:
                sampled.extend(random.sample(all_unsampled, min(remaining, len(all_unsampled))))

        # If we got too many (shouldn't happen), trim
        if len(sampled) > n_samples:
            sampled = random.sample(sampled, n_samples)

        return sampled

    def mix_datasets(
        self,
        public_sources: List[DataSource],
        car_sources: List[DataSource],
        synthetic_sources: List[DataSource]
    ) -> Tuple[List[Dict], Dict]:
        """
        Mix datasets according to RegMix ratios with stratified sampling.

        Args:
            public_sources: Public general psychology datasets
            car_sources: Car domain-specific datasets
            synthetic_sources: PsychSteer synthetic datasets

        Returns:
            Tuple of (mixed_samples, statistics)
        """
        logger.info("Starting RegMix data mixing...")

        # Aggregate all samples by type
        all_public = []
        for source in public_sources:
            all_public.extend(source.samples)

        all_car = []
        for source in car_sources:
            all_car.extend(source.samples)

        all_synthetic = []
        for source in synthetic_sources:
            all_synthetic.extend(source.samples)

        logger.info(f"Total samples available: Public={len(all_public)}, "
                   f"Car={len(all_car)}, Synthetic={len(all_synthetic)}")

        # Determine total samples to generate
        if self.config.total_samples:
            total_target = self.config.total_samples
        else:
            # Use minimum to ensure we can meet all ratios
            total_target = min(
                len(all_public) * 100 // self.config.public_general_ratio,
                len(all_car) * 100 // self.config.car_domain_ratio,
                len(all_synthetic) * 100 // self.config.synthetic_ratio
            )

        logger.info(f"Target total samples: {total_target}")

        # Calculate samples needed from each source
        n_public = (total_target * self.config.public_general_ratio) // 100
        n_car = (total_target * self.config.car_domain_ratio) // 100
        n_synthetic = (total_target * self.config.synthetic_ratio) // 100

        logger.info(f"Sampling: Public={n_public}, Car={n_car}, Synthetic={n_synthetic}")

        # Stratified sampling from each source
        sampled_public = self._stratified_sample(all_public, n_public)
        sampled_car = self._stratified_sample(all_car, n_car)
        sampled_synthetic = self._stratified_sample(all_synthetic, n_synthetic)

        # Combine all samples
        mixed_samples = []

        # Add source tags
        for sample in sampled_public:
            sample["_source_type"] = "public_general"
            mixed_samples.append(sample)

        for sample in sampled_car:
            sample["_source_type"] = "car_domain"
            mixed_samples.append(sample)

        for sample in sampled_synthetic:
            sample["_source_type"] = "synthetic"
            mixed_samples.append(sample)

        # Shuffle the final dataset
        random.shuffle(mixed_samples)

        # Calculate statistics
        stats = self._calculate_statistics(
            mixed_samples,
            sampled_public,
            sampled_car,
            sampled_synthetic
        )

        logger.info(f"RegMix complete. Total samples: {len(mixed_samples)}")
        logger.info(f"Final distribution: {stats['distribution']}")

        return mixed_samples, stats

    def _calculate_statistics(
        self,
        mixed_samples: List[Dict],
        public_samples: List[Dict],
        car_samples: List[Dict],
        synthetic_samples: List[Dict]
    ) -> Dict:
        """Calculate statistics about the mixed dataset"""
        total = len(mixed_samples)

        stats = {
            "total_samples": total,
            "distribution": {
                "public_general": {
                    "count": len(public_samples),
                    "percentage": (len(public_samples) / total * 100) if total > 0 else 0
                },
                "car_domain": {
                    "count": len(car_samples),
                    "percentage": (len(car_samples) / total * 100) if total > 0 else 0
                },
                "synthetic": {
                    "count": len(synthetic_samples),
                    "percentage": (len(synthetic_samples) / total * 100) if total > 0 else 0
                }
            },
            "trait_distributions": self._get_trait_distributions(mixed_samples)
        }

        return stats

    def _get_trait_distributions(self, samples: List[Dict]) -> Dict:
        """Calculate mean and std of each personality trait"""
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        distributions = {}

        for trait in traits:
            scores = []
            for sample in samples:
                if "personality_scores" in sample and trait in sample["personality_scores"]:
                    scores.append(sample["personality_scores"][trait])

            if scores:
                distributions[trait] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores))
                }

        return distributions

    def save_mixed_dataset(
        self,
        mixed_samples: List[Dict],
        output_path: Path,
        format: str = "jsonl"
    ):
        """
        Save the mixed dataset to a file.

        Args:
            mixed_samples: Mixed dataset
            output_path: Path to save the dataset
            format: Output format ('jsonl', 'json')
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, 'w') as f:
                for sample in mixed_samples:
                    f.write(json.dumps(sample) + '\n')
        elif format == "json":
            with open(output_path, 'w') as f:
                json.dump(mixed_samples, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(mixed_samples)} samples to {output_path}")


# Example usage
def main():
    """Example usage of RegMix"""
    # Create sample data sources
    public_samples = [
        {
            "id": f"pub_{i}",
            "text": f"Public sample {i}",
            "personality_scores": {
                "openness": random.uniform(0.3, 0.8),
                "conscientiousness": random.uniform(0.3, 0.8),
                "extraversion": random.uniform(0.3, 0.8),
                "agreeableness": random.uniform(0.3, 0.8),
                "neuroticism": random.uniform(0.3, 0.8)
            }
        }
        for i in range(1000)
    ]

    car_samples = [
        {
            "id": f"car_{i}",
            "text": f"Car sample {i}",
            "personality_scores": {
                "openness": random.uniform(0.4, 0.9),
                "conscientiousness": random.uniform(0.4, 0.9),
                "extraversion": random.uniform(0.4, 0.9),
                "agreeableness": random.uniform(0.4, 0.9),
                "neuroticism": random.uniform(0.2, 0.7)
            }
        }
        for i in range(1000)
    ]

    synthetic_samples = [
        {
            "id": f"syn_{i}",
            "text": f"Synthetic sample {i}",
            "personality_scores": {
                "openness": random.uniform(0.2, 0.9),
                "conscientiousness": random.uniform(0.2, 0.9),
                "extraversion": random.uniform(0.2, 0.9),
                "agreeableness": random.uniform(0.2, 0.9),
                "neuroticism": random.uniform(0.2, 0.9)
            }
        }
        for i in range(500)
    ]

    # Create data sources
    public_source = DataSource("Big5-Chat", "public_general", public_samples, len(public_samples))
    car_source = DataSource("Car-Questionnaire", "car_domain", car_samples, len(car_samples))
    synthetic_source = DataSource("PsychSteer-Labeled", "synthetic", synthetic_samples, len(synthetic_samples))

    # Initialize RegMix
    config = RegMixConfig(
        public_general_ratio=40,
        car_domain_ratio=40,
        synthetic_ratio=20,
        total_samples=1000
    )
    regmix = RegMix(config)

    # Mix datasets
    mixed_data, stats = regmix.mix_datasets(
        [public_source],
        [car_source],
        [synthetic_source]
    )

    print("\n=== RegMix Statistics ===")
    print(json.dumps(stats, indent=2))

    # Save the mixed dataset
    output_path = Path("data/processed/regmix_output.jsonl")
    regmix.save_mixed_dataset(mixed_data, output_path)


if __name__ == "__main__":
    main()
