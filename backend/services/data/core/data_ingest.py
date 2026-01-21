"""
Data Ingestion Pipeline for Car-Psycho Platform

This module orchestrates the complete data pipeline:
1. Download public datasets (Big5-Chat, PANDORA)
2. Process proprietary Car Needs Questionnaire
3. Apply PsychSteer labeling to car data
4. Apply RegMix to create the final training dataset

Entry point for Phase 1 of the platform.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

from .psychsteer import PsychSteer, CarProfileInput, TeacherModel
from .regmix import RegMix, RegMixConfig, DataSource
from .downloaders import DatasetDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataPipelineConfig:
    """Configuration for the complete data pipeline"""
    # Paths
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    car_questionnaire_dir: Path = Path("data/car_questionnaire")
    synthetic_data_dir: Path = Path("data/synthetic")
    output_dir: Path = Path("data/processed")

    # Dataset sources
    download_big5_chat: bool = True
    download_pandora: bool = True

    # PsychSteer configuration
    teacher_model: str = "gpt-4o"
    psychsteer_batch_size: int = 10
    psychsteer_temperature: float = 0.3

    # RegMix configuration
    regmix_public_ratio: int = 40
    regmix_car_ratio: int = 40
    regmix_synthetic_ratio: int = 20
    regmix_total_samples: Optional[int] = None

    # Output configuration
    output_format: str = "jsonl"
    output_filename: str = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


class DataPipeline:
    """
    Main data ingestion pipeline orchestrator.

    This class coordinates:
    1. Public dataset downloads
    2. Car questionnaire processing
    3. PsychSteer synthetic labeling
    4. RegMix data mixing
    """

    def __init__(self, config: DataPipelineConfig = None):
        """
        Initialize the data pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or DataPipelineConfig()

        # Create directories
        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.config.car_questionnaire_dir.mkdir(parents=True, exist_ok=True)
        self.config.synthetic_data_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Data pipeline initialized")

    async def download_public_datasets(self) -> Dict[str, Path]:
        """
        Download and process public datasets.

        Returns:
            Dictionary mapping dataset names to file paths
        """
        logger.info("=== Step 1: Downloading Public Datasets ===")

        downloader = DatasetDownloader(self.config.raw_data_dir)
        downloaded = {}

        if self.config.download_big5_chat:
            logger.info("Downloading Big5-Chat dataset...")
            big5_path = await downloader.download_big5_chat()
            if big5_path:
                downloaded["big5_chat"] = big5_path

        if self.config.download_pandora:
            logger.info("Downloading PANDORA dataset...")
            pandora_path = await downloader.download_pandora()
            if pandora_path:
                downloaded["pandora"] = pandora_path

        logger.info(f"Downloaded {len(downloaded)} public datasets")
        return downloaded

    def load_car_questionnaires(self) -> List[CarProfileInput]:
        """
        Load proprietary car questionnaire data.

        Returns:
            List of CarProfileInput objects
        """
        logger.info("=== Step 2: Loading Car Questionnaires ===")

        car_profiles = []
        questionnaire_files = list(self.config.car_questionnaire_dir.glob("*.json"))

        if not questionnaire_files:
            logger.warning(f"No questionnaire files found in {self.config.car_questionnaire_dir}")
            logger.warning("Creating sample questionnaire for demonstration...")
            self._create_sample_questionnaires()
            questionnaire_files = list(self.config.car_questionnaire_dir.glob("*.json"))

        for file_path in questionnaire_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                profile = CarProfileInput(
                    customer_id=data.get("customer_id"),
                    responses=data.get("responses", {}),
                    raw_text=data.get("raw_text")
                )
                car_profiles.append(profile)

            except Exception as e:
                logger.error(f"Failed to load questionnaire from {file_path}: {e}")

        logger.info(f"Loaded {len(car_profiles)} car questionnaires")
        return car_profiles

    def _create_sample_questionnaires(self):
        """Create sample questionnaire files for demonstration"""
        samples = [
            {
                "customer_id": "SAMPLE_001",
                "responses": {
                    "Budget Range": "$35,000 - $45,000",
                    "Primary Use": "Daily commute to work (30 miles) and weekend family trips",
                    "Family Size": "4 people (2 adults, 2 children ages 5 and 8)",
                    "Key Priorities": "Safety ratings are my top concern. I want excellent crash test scores, modern safety features like automatic emergency braking, lane departure warning. Also important: reliability and low maintenance costs.",
                    "Aesthetic Preference": "I prefer a practical, modern design. Nothing too flashy or sporty. Clean lines, professional look that will age well.",
                    "Technology Interest": "I want essential safety tech like backup camera, blind spot monitoring, and adaptive cruise control. Not interested in cutting-edge experimental features.",
                    "Environmental Concern": "Moderately important. I'm considering hybrid options to save on gas and reduce emissions, but pure electric is too limiting for now.",
                    "Maintenance Attitude": "I want a solid warranty (at least 5 years) and a car known for reliability. I'm not handy with cars, so I need something that won't need frequent repairs.",
                    "Decision Timeline": "Planning to buy within 2 months. I've been researching for a while and feel ready to make a decision."
                },
                "raw_text": "I'm a parent focused on keeping my family safe. I've done a lot of research on safety ratings and want a car that will protect my kids. Reliability is also key because I don't want to deal with breakdowns or expensive repairs."
            },
            {
                "customer_id": "SAMPLE_002",
                "responses": {
                    "Budget Range": "$50,000 - $70,000",
                    "Primary Use": "Daily driving in the city and weekend road trips",
                    "Family Size": "2 people (couple, no children)",
                    "Key Priorities": "Performance and style are what I care about most. I want something that turns heads and is fun to drive. Brand prestige matters to me.",
                    "Aesthetic Preference": "Sporty, aggressive design. I love bold colors, sleek lines, and a commanding presence on the road. Want something that makes a statement.",
                    "Technology Interest": "Give me everything - the latest infotainment system, premium sound, connected features, performance displays. I want to feel like I'm driving the future.",
                    "Environmental Concern": "Not a priority for me. If it's fun to drive, that's what matters.",
                    "Maintenance Attitude": "I'll pay for premium service. I want the dealership experience and don't mind higher maintenance costs for a quality vehicle.",
                    "Decision Timeline": "Looking to buy soon, but not rushed. Want to test drive several high-performance options."
                },
                "raw_text": "I work hard and I want a car that reflects my success. Something sporty, powerful, and eye-catching. I'm drawn to brands with racing heritage and performance credentials."
            },
            {
                "customer_id": "SAMPLE_003",
                "responses": {
                    "Budget Range": "$25,000 - $32,000",
                    "Primary Use": "Work commute and errands around town",
                    "Family Size": "1 person (single)",
                    "Key Priorities": "Fuel efficiency and overall cost of ownership. I want the best value for my money and lowest running costs.",
                    "Aesthetic Preference": "I'm not picky about looks. Function over form. Just want something clean and simple that doesn't stand out.",
                    "Technology Interest": "Basic features are fine. I need Bluetooth for phone calls and maybe a backup camera. Don't need fancy bells and whistles.",
                    "Environmental Concern": "Very important. I'm seriously considering a fully electric or plug-in hybrid to reduce my carbon footprint and save on gas.",
                    "Maintenance Attitude": "I plan to do basic maintenance myself where possible. Want a car known for being easy to maintain and with widely available parts.",
                    "Decision Timeline": "Not in a rush. Will wait for the right deal and maybe look at certified pre-owned options."
                },
                "raw_text": "I'm practical and environmentally conscious. I don't need anything flashy, just reliable, efficient transportation that aligns with my values."
            }
        ]

        for sample in samples:
            file_path = self.config.car_questionnaire_dir / f"{sample['customer_id']}.json"
            with open(file_path, 'w') as f:
                json.dump(sample, f, indent=2)

        logger.info(f"Created {len(samples)} sample questionnaires")

    async def apply_psychsteer_labeling(
        self,
        car_profiles: List[CarProfileInput]
    ) -> List[Dict]:
        """
        Apply PsychSteer labeling to car questionnaire data.

        Args:
            car_profiles: List of car profile inputs

        Returns:
            List of labeled samples with personality scores
        """
        logger.info("=== Step 3: Applying PsychSteer Labeling ===")

        # Initialize PsychSteer
        teacher_model_map = {
            "gpt-4o": TeacherModel.GPT4O,
            "gpt-4-turbo": TeacherModel.GPT4_TURBO,
            "claude-3-5-sonnet-20241022": TeacherModel.CLAUDE_SONNET,
            "claude-opus-4-5-20251101": TeacherModel.CLAUDE_OPUS
        }

        teacher_model = teacher_model_map.get(
            self.config.teacher_model,
            TeacherModel.GPT4O
        )

        psychsteer = PsychSteer(
            teacher_model=teacher_model,
            temperature=self.config.psychsteer_temperature
        )

        # Batch label the profiles
        labeled_results = await psychsteer.batch_label(
            car_profiles,
            batch_size=self.config.psychsteer_batch_size
        )

        # Convert to training format
        training_samples = []
        for result in labeled_results:
            sample = {
                "id": result["customer_id"],
                "text": self._format_questionnaire_as_text(result),
                "personality_scores": result["personality_scores"],
                "confidence_scores": result.get("confidence_scores", {}),
                "reasoning": result.get("reasoning", {}),
                "source": "psychsteer_car_questionnaire",
                "teacher_model": result["teacher_model"]
            }
            training_samples.append(sample)

        # Save synthetic data
        synthetic_output = self.config.synthetic_data_dir / "psychsteer_labeled.jsonl"
        with open(synthetic_output, 'w') as f:
            for sample in training_samples:
                f.write(json.dumps(sample) + '\n')

        logger.info(f"PsychSteer labeled {len(training_samples)} samples")
        logger.info(f"Saved to {synthetic_output}")

        return training_samples

    def _format_questionnaire_as_text(self, labeled_result: Dict) -> str:
        """Format questionnaire data as conversational text for training"""
        # This creates a natural language representation of the questionnaire
        # that the model can learn from
        text = "Customer: I'm looking for help choosing a car.\n\n"

        if "reasoning" in labeled_result:
            for trait, reasoning in labeled_result["reasoning"].items():
                text += f"Based on the responses, the customer shows {reasoning}\n"

        return text

    def load_public_datasets(self, downloaded_paths: Dict[str, Path]) -> List[DataSource]:
        """
        Load and parse public datasets into DataSource objects.

        Args:
            downloaded_paths: Dictionary of dataset names to paths

        Returns:
            List of DataSource objects
        """
        logger.info("=== Step 4: Loading Public Datasets ===")

        data_sources = []

        for dataset_name, file_path in downloaded_paths.items():
            try:
                samples = []

                if file_path.suffix == ".jsonl":
                    with open(file_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                samples.append(json.loads(line))
                elif file_path.suffix == ".json":
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            samples = data
                        else:
                            samples = [data]

                source = DataSource(
                    name=dataset_name,
                    source_type="public_general",
                    samples=samples,
                    num_samples=len(samples)
                )
                data_sources.append(source)

                logger.info(f"Loaded {len(samples)} samples from {dataset_name}")

            except Exception as e:
                logger.error(f"Failed to load {dataset_name} from {file_path}: {e}")

        return data_sources

    def apply_regmix(
        self,
        public_sources: List[DataSource],
        car_sources: List[DataSource],
        synthetic_sources: List[DataSource]
    ) -> tuple[List[Dict], Dict]:
        """
        Apply RegMix to create the final training dataset.

        Args:
            public_sources: Public dataset sources
            car_sources: Car domain sources
            synthetic_sources: Synthetic labeled sources

        Returns:
            Tuple of (mixed_samples, statistics)
        """
        logger.info("=== Step 5: Applying RegMix ===")

        config = RegMixConfig(
            public_general_ratio=self.config.regmix_public_ratio,
            car_domain_ratio=self.config.regmix_car_ratio,
            synthetic_ratio=self.config.regmix_synthetic_ratio,
            total_samples=self.config.regmix_total_samples
        )

        regmix = RegMix(config)
        mixed_samples, stats = regmix.mix_datasets(
            public_sources,
            car_sources,
            synthetic_sources
        )

        return mixed_samples, stats

    async def run_full_pipeline(self) -> Dict:
        """
        Run the complete data ingestion pipeline.

        Returns:
            Dictionary with pipeline results and statistics
        """
        logger.info("=" * 60)
        logger.info("Starting Full Data Pipeline")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Step 1: Download public datasets
        downloaded_paths = await self.download_public_datasets()
        public_sources = self.load_public_datasets(downloaded_paths)

        # Step 2: Load car questionnaires
        car_profiles = self.load_car_questionnaires()

        # Step 3: Apply PsychSteer labeling
        if car_profiles:
            synthetic_samples = await self.apply_psychsteer_labeling(car_profiles)
            synthetic_source = DataSource(
                name="psychsteer_labeled",
                source_type="synthetic",
                samples=synthetic_samples,
                num_samples=len(synthetic_samples)
            )
            synthetic_sources = [synthetic_source]
        else:
            logger.warning("No car profiles to label, skipping PsychSteer")
            synthetic_sources = []

        # For demonstration, if we have no car domain data, use a subset of public data
        if not car_profiles:
            logger.warning("No car domain data, using public data subset for car domain")
            if public_sources and len(public_sources[0].samples) > 100:
                car_subset = public_sources[0].samples[:100]
                car_source = DataSource(
                    name="car_domain_placeholder",
                    source_type="car_domain",
                    samples=car_subset,
                    num_samples=len(car_subset)
                )
                car_sources = [car_source]
            else:
                car_sources = []
        else:
            car_sources = []  # Will be populated from processed questionnaires

        # Step 4: Apply RegMix
        if public_sources and (car_sources or synthetic_sources):
            mixed_samples, regmix_stats = self.apply_regmix(
                public_sources,
                car_sources or [DataSource("empty_car", "car_domain", [], 0)],
                synthetic_sources or [DataSource("empty_synthetic", "synthetic", [], 0)]
            )

            # Step 5: Save final training dataset
            output_path = self.config.output_dir / self.config.output_filename
            regmix = RegMix()
            regmix.save_mixed_dataset(
                mixed_samples,
                output_path,
                format=self.config.output_format
            )

            logger.info(f"Final training dataset saved to: {output_path}")
        else:
            logger.error("Insufficient data sources for RegMix")
            mixed_samples = []
            regmix_stats = {}

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Compile results
        results = {
            "status": "success" if mixed_samples else "partial",
            "duration_seconds": duration,
            "public_datasets": len(public_sources),
            "car_profiles": len(car_profiles),
            "synthetic_samples": len(synthetic_sources[0].samples) if synthetic_sources else 0,
            "final_dataset_size": len(mixed_samples),
            "output_path": str(self.config.output_dir / self.config.output_filename),
            "regmix_stats": regmix_stats
        }

        logger.info("=" * 60)
        logger.info("Pipeline Complete!")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Final dataset: {len(mixed_samples)} samples")
        logger.info("=" * 60)

        return results


# CLI interface
async def main():
    """Main entry point for data pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Car-Psycho Data Ingestion Pipeline")
    parser.add_argument("--teacher-model", default="gpt-4o", help="Teacher model for PsychSteer")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading public datasets")
    parser.add_argument("--total-samples", type=int, help="Total samples for RegMix")
    parser.add_argument("--output", default=None, help="Output filename")

    args = parser.parse_args()

    config = DataPipelineConfig(
        download_big5_chat=not args.no_download,
        download_pandora=not args.no_download,
        teacher_model=args.teacher_model,
        regmix_total_samples=args.total_samples
    )

    if args.output:
        config.output_filename = args.output

    pipeline = DataPipeline(config)
    results = await pipeline.run_full_pipeline()

    print("\n=== Pipeline Results ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())