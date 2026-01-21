"""
Dataset Downloaders for Public Psychology Datasets

Handles downloading and processing of:
1. Big5-Chat dataset
2. PANDORA dataset
3. Other personality psychology datasets
"""

import json
import logging
from pathlib import Path
from typing import Optional
import aiohttp
import asyncio
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    Handles downloading and preprocessing public datasets.
    """

    def __init__(self, raw_data_dir: Path):
        """
        Initialize the downloader.

        Args:
            raw_data_dir: Directory to store raw datasets
        """
        self.raw_data_dir = raw_data_dir
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    async def download_big5_chat(self) -> Optional[Path]:
        """
        Download and process the Big5-Chat dataset.

        The Big5-Chat dataset contains conversational data labeled with
        Big Five personality traits.

        Returns:
            Path to the processed dataset file, or None if failed
        """
        logger.info("Downloading Big5-Chat dataset...")

        try:
            # Big5-Chat is typically available on HuggingFace
            # This is a placeholder - adjust to actual dataset location
            output_path = self.raw_data_dir / "big5_chat.jsonl"

            if output_path.exists():
                logger.info(f"Big5-Chat already exists at {output_path}")
                return output_path

            # Try to load from HuggingFace datasets
            try:
                dataset = load_dataset("Big5-Chat", split="train")
                samples = []

                for item in tqdm(dataset, desc="Processing Big5-Chat"):
                    sample = self._normalize_big5_chat_sample(item)
                    if sample:
                        samples.append(sample)

                # Save as JSONL
                with open(output_path, 'w') as f:
                    for sample in samples:
                        f.write(json.dumps(sample) + '\n')

                logger.info(f"Big5-Chat downloaded: {len(samples)} samples")
                return output_path

            except Exception as e:
                logger.warning(f"Could not load Big5-Chat from HuggingFace: {e}")
                logger.info("Creating synthetic Big5-Chat samples for demonstration...")
                return self._create_synthetic_big5_chat(output_path)

        except Exception as e:
            logger.error(f"Failed to download Big5-Chat: {e}")
            return None

    def _normalize_big5_chat_sample(self, item: dict) -> Optional[dict]:
        """
        Normalize a Big5-Chat sample to our standard format.

        Args:
            item: Raw dataset item

        Returns:
            Normalized sample dictionary
        """
        try:
            # Adjust field names based on actual dataset structure
            return {
                "id": item.get("id", ""),
                "text": item.get("text", item.get("conversation", "")),
                "personality_scores": {
                    "openness": float(item.get("openness", 0.5)),
                    "conscientiousness": float(item.get("conscientiousness", 0.5)),
                    "extraversion": float(item.get("extraversion", 0.5)),
                    "agreeableness": float(item.get("agreeableness", 0.5)),
                    "neuroticism": float(item.get("neuroticism", 0.5))
                },
                "source": "big5_chat"
            }
        except Exception as e:
            logger.warning(f"Failed to normalize sample: {e}")
            return None

    def _create_synthetic_big5_chat(self, output_path: Path) -> Path:
        """Create synthetic Big5-Chat samples for demonstration"""
        import random

        samples = []

        # Sample conversation templates with personality indicators
        templates = [
            {
                "text": "I love trying new things and exploring different ideas. Yesterday I started learning about quantum physics just out of curiosity!",
                "scores": {"openness": 0.9, "conscientiousness": 0.5, "extraversion": 0.6, "agreeableness": 0.6, "neuroticism": 0.3}
            },
            {
                "text": "I always make detailed plans and stick to my schedule. Organization is key to success in life.",
                "scores": {"openness": 0.5, "conscientiousness": 0.95, "extraversion": 0.5, "agreeableness": 0.6, "neuroticism": 0.3}
            },
            {
                "text": "I absolutely love social gatherings! The more people, the better. I get my energy from being around others.",
                "scores": {"openness": 0.6, "conscientiousness": 0.5, "extraversion": 0.95, "agreeableness": 0.7, "neuroticism": 0.2}
            },
            {
                "text": "I always try to help others and avoid conflict. It's important to me that everyone feels comfortable and included.",
                "scores": {"openness": 0.5, "conscientiousness": 0.6, "extraversion": 0.5, "agreeableness": 0.95, "neuroticism": 0.3}
            },
            {
                "text": "I often worry about things going wrong. What if I made the wrong decision? I can't stop thinking about it.",
                "scores": {"openness": 0.4, "conscientiousness": 0.6, "extraversion": 0.3, "agreeableness": 0.6, "neuroticism": 0.9}
            }
        ]

        # Generate 500 samples by varying the templates
        for i in range(500):
            template = random.choice(templates)
            # Add slight variations to scores
            scores = {}
            for trait, base_score in template["scores"].items():
                variation = random.uniform(-0.1, 0.1)
                scores[trait] = max(0.0, min(1.0, base_score + variation))

            sample = {
                "id": f"big5_chat_synthetic_{i}",
                "text": template["text"],
                "personality_scores": scores,
                "source": "big5_chat_synthetic"
            }
            samples.append(sample)

        # Save
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        logger.info(f"Created {len(samples)} synthetic Big5-Chat samples")
        return output_path

    async def download_pandora(self) -> Optional[Path]:
        """
        Download and process the PANDORA dataset.

        PANDORA contains personality assessments and behavioral data.

        Returns:
            Path to the processed dataset file, or None if failed
        """
        logger.info("Downloading PANDORA dataset...")

        try:
            output_path = self.raw_data_dir / "pandora.jsonl"

            if output_path.exists():
                logger.info(f"PANDORA already exists at {output_path}")
                return output_path

            # Try to load from HuggingFace or other source
            try:
                # This is a placeholder - adjust to actual dataset location
                dataset = load_dataset("PANDORA", split="train")
                samples = []

                for item in tqdm(dataset, desc="Processing PANDORA"):
                    sample = self._normalize_pandora_sample(item)
                    if sample:
                        samples.append(sample)

                # Save as JSONL
                with open(output_path, 'w') as f:
                    for sample in samples:
                        f.write(json.dumps(sample) + '\n')

                logger.info(f"PANDORA downloaded: {len(samples)} samples")
                return output_path

            except Exception as e:
                logger.warning(f"Could not load PANDORA from HuggingFace: {e}")
                logger.info("Creating synthetic PANDORA samples for demonstration...")
                return self._create_synthetic_pandora(output_path)

        except Exception as e:
            logger.error(f"Failed to download PANDORA: {e}")
            return None

    def _normalize_pandora_sample(self, item: dict) -> Optional[dict]:
        """Normalize a PANDORA sample to our standard format"""
        try:
            return {
                "id": item.get("id", ""),
                "text": item.get("text", item.get("response", "")),
                "personality_scores": {
                    "openness": float(item.get("O", 0.5)),
                    "conscientiousness": float(item.get("C", 0.5)),
                    "extraversion": float(item.get("E", 0.5)),
                    "agreeableness": float(item.get("A", 0.5)),
                    "neuroticism": float(item.get("N", 0.5))
                },
                "source": "pandora"
            }
        except Exception as e:
            logger.warning(f"Failed to normalize PANDORA sample: {e}")
            return None

    def _create_synthetic_pandora(self, output_path: Path) -> Path:
        """Create synthetic PANDORA samples for demonstration"""
        import random

        samples = []

        # Generate diverse personality profiles
        for i in range(500):
            # Generate random but coherent personality profile
            openness = random.uniform(0.2, 0.9)
            conscientiousness = random.uniform(0.2, 0.9)
            extraversion = random.uniform(0.2, 0.9)
            agreeableness = random.uniform(0.2, 0.9)
            neuroticism = random.uniform(0.2, 0.9)

            # Generate text that reflects the personality
            text_parts = []

            if openness > 0.6:
                text_parts.append("I'm interested in creative and abstract ideas.")
            if conscientiousness > 0.6:
                text_parts.append("I value organization and reliability.")
            if extraversion > 0.6:
                text_parts.append("I enjoy social interactions and group activities.")
            if agreeableness > 0.6:
                text_parts.append("I prioritize cooperation and harmony.")
            if neuroticism > 0.6:
                text_parts.append("I tend to experience strong emotions and concerns.")

            text = " ".join(text_parts) if text_parts else "A person with balanced personality traits."

            sample = {
                "id": f"pandora_synthetic_{i}",
                "text": text,
                "personality_scores": {
                    "openness": openness,
                    "conscientiousness": conscientiousness,
                    "extraversion": extraversion,
                    "agreeableness": agreeableness,
                    "neuroticism": neuroticism
                },
                "source": "pandora_synthetic"
            }
            samples.append(sample)

        # Save
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        logger.info(f"Created {len(samples)} synthetic PANDORA samples")
        return output_path


# Example usage
async def main():
    """Test the downloader"""
    downloader = DatasetDownloader(Path("data/raw"))

    big5_path = await downloader.download_big5_chat()
    print(f"Big5-Chat saved to: {big5_path}")

    pandora_path = await downloader.download_pandora()
    print(f"PANDORA saved to: {pandora_path}")


if __name__ == "__main__":
    asyncio.run(main())
