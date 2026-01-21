"""
PsychSteer: Synthetic Labeling System for Car Sales Domain

This module implements the PsychSteer methodology from "Architecting the Artificial Psychologist"
adapted for the car sales domain. It uses a teacher LLM (GPT-4o or Claude) to label customer
car needs questionnaire responses with Big Five personality traits.

Key Innovation:
- Domain Transfer: Maps car preferences → personality traits
- Research-Based Mappings: Uses validated psychological correlations
- Teacher Model: Uses advanced LLM to generate high-quality labels
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from dataclasses import dataclass
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeacherModel(Enum):
    """Supported teacher models for PsychSteer"""
    GPT4O = "gpt-4o"
    GPT4_TURBO = "gpt-4-turbo"
    CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_OPUS = "claude-opus-4-5-20251101"


@dataclass
class PersonalityScores:
    """Big Five personality scores (0.0 to 1.0)"""
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism
        }

    def validate(self) -> bool:
        """Ensure all scores are in valid range"""
        return all(0.0 <= score <= 1.0 for score in self.to_dict().values())


@dataclass
class CarProfileInput:
    """Input data structure for car questionnaire"""
    customer_id: Optional[str]
    responses: Dict[str, str]  # question_id -> answer
    raw_text: Optional[str]  # free-form text if available


class PsychSteer:
    """
    PsychSteer implementation for car sales domain.

    This class implements the synthetic labeling pipeline that uses a teacher LLM
    to label car preference data with Big Five personality traits.
    """

    # Research-based prompt for personality inference from car preferences
    SYSTEM_PROMPT = """You are an expert psychologist specializing in personality assessment and consumer behavior.

Your task is to analyze customer responses about their car preferences and needs, then infer their Big Five personality traits.

The Big Five (OCEAN) personality model consists of:
1. **Openness**: Creativity, curiosity, preference for variety vs. routine
2. **Conscientiousness**: Organization, dependability, discipline, planning
3. **Extraversion**: Sociability, assertiveness, energy, enthusiasm
4. **Agreeableness**: Compassion, cooperation, trust, politeness
5. **Neuroticism**: Emotional instability, anxiety, worry, stress

# Research-Based Mappings for Car Preferences:

**High Openness (0.7-1.0):**
- Interest in innovative features, new technology
- Aesthetic uniqueness, unconventional designs
- Electric/hybrid vehicles, experimental models
- Emphasis on "experience" over practicality

**High Conscientiousness (0.7-1.0):**
- Safety ratings and reliability as top priorities
- Detailed maintenance plans and warranties
- Fuel efficiency and cost-effectiveness
- Well-researched, methodical decision-making

**High Extraversion (0.7-1.0):**
- Sporty, attention-grabbing vehicles
- Performance and speed preferences
- Social features (passenger space, entertainment)
- Brand prestige and status

**High Agreeableness (0.7-1.0):**
- Family-oriented features
- Comfort and passenger well-being
- Environmental considerations
- Affordable, "fair value" expectations

**High Neuroticism (0.7-1.0):**
- Excessive concern about safety
- Worry about maintenance costs, breakdowns
- Need for reassurance and guarantees
- Indecisiveness or anxiety in responses

# Instructions:
Analyze the customer's car questionnaire responses and provide:
1. Big Five scores (0.0 to 1.0 for each trait)
2. Confidence scores for each trait (0.0 to 1.0)
3. Brief reasoning for each score

Return your analysis as a JSON object with this exact structure:
{
    "personality_scores": {
        "openness": 0.75,
        "conscientiousness": 0.85,
        "extraversion": 0.60,
        "agreeableness": 0.70,
        "neuroticism": 0.40
    },
    "confidence_scores": {
        "openness": 0.80,
        "conscientiousness": 0.90,
        "extraversion": 0.75,
        "agreeableness": 0.85,
        "neuroticism": 0.70
    },
    "reasoning": {
        "openness": "Customer shows interest in electric vehicles and new technology features, indicating curiosity and openness to new experiences.",
        "conscientiousness": "Strong emphasis on safety ratings, reliability, and detailed research suggests high conscientiousness.",
        "extraversion": "Moderate social features interest, but not strongly focused on status or performance.",
        "agreeableness": "Family-oriented priorities and environmental concerns indicate high agreeableness.",
        "neuroticism": "Measured concerns about safety without excessive anxiety; shows balanced decision-making."
    }
}

Be precise, use the full 0.0-1.0 range, and base your scores on psychological research."""

    def __init__(
        self,
        teacher_model: TeacherModel = TeacherModel.GPT4O,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        temperature: float = 0.3,  # Lower temperature for more consistent labeling
        max_retries: int = 3
    ):
        """
        Initialize PsychSteer labeler.

        Args:
            teacher_model: Which teacher model to use
            openai_api_key: OpenAI API key (if using GPT models)
            anthropic_api_key: Anthropic API key (if using Claude models)
            temperature: Sampling temperature for the teacher model
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize API clients
        if teacher_model in [TeacherModel.GPT4O, TeacherModel.GPT4_TURBO]:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required for GPT models")
            self.openai_client = AsyncOpenAI(api_key=api_key)
        else:
            api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key required for Claude models")
            self.anthropic_client = AsyncAnthropic(api_key=api_key)

        logger.info(f"PsychSteer initialized with teacher model: {teacher_model.value}")

    def _format_questionnaire_for_prompt(self, car_profile: CarProfileInput) -> str:
        """Format the questionnaire data into a clear prompt for the teacher model"""
        formatted_text = "# Customer Car Questionnaire Responses:\n\n"

        for question, answer in car_profile.responses.items():
            formatted_text += f"**{question}:** {answer}\n\n"

        if car_profile.raw_text:
            formatted_text += f"\n# Additional Comments:\n{car_profile.raw_text}\n"

        return formatted_text

    async def _call_openai(self, user_prompt: str) -> str:
        """Call OpenAI API"""
        for attempt in range(self.max_retries):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=self.teacher_model.value,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _call_anthropic(self, user_prompt: str) -> str:
        """Call Anthropic API"""
        for attempt in range(self.max_retries):
            try:
                response = await self.anthropic_client.messages.create(
                    model=self.teacher_model.value,
                    max_tokens=2048,
                    temperature=self.temperature,
                    system=self.SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"Anthropic API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    def _parse_teacher_response(self, response_text: str) -> Tuple[PersonalityScores, Dict, Dict]:
        """
        Parse the teacher model's JSON response into structured data.

        Returns:
            Tuple of (PersonalityScores, confidence_dict, reasoning_dict)
        """
        try:
            # Try to parse JSON
            data = json.loads(response_text)

            personality_scores = PersonalityScores(
                openness=float(data["personality_scores"]["openness"]),
                conscientiousness=float(data["personality_scores"]["conscientiousness"]),
                extraversion=float(data["personality_scores"]["extraversion"]),
                agreeableness=float(data["personality_scores"]["agreeableness"]),
                neuroticism=float(data["personality_scores"]["neuroticism"])
            )

            if not personality_scores.validate():
                raise ValueError("Personality scores out of valid range [0.0, 1.0]")

            confidence_scores = data.get("confidence_scores", {})
            reasoning = data.get("reasoning", {})

            return personality_scores, confidence_scores, reasoning

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse teacher response: {e}")
            logger.error(f"Response text: {response_text}")
            raise ValueError(f"Invalid response format from teacher model: {e}")

    async def label_car_profile(
        self,
        car_profile: CarProfileInput
    ) -> Dict:
        """
        Main method: Label a car questionnaire with personality traits.

        Args:
            car_profile: Customer car questionnaire data

        Returns:
            Dictionary with personality scores, confidence, and reasoning
        """
        logger.info(f"Labeling car profile for customer: {car_profile.customer_id}")

        # Format the input for the teacher model
        user_prompt = self._format_questionnaire_for_prompt(car_profile)

        # Call the teacher model
        if self.teacher_model in [TeacherModel.GPT4O, TeacherModel.GPT4_TURBO]:
            response_text = await self._call_openai(user_prompt)
        else:
            response_text = await self._call_anthropic(user_prompt)

        # Parse the response
        personality_scores, confidence_scores, reasoning = self._parse_teacher_response(response_text)

        result = {
            "customer_id": car_profile.customer_id,
            "personality_scores": personality_scores.to_dict(),
            "confidence_scores": confidence_scores,
            "reasoning": reasoning,
            "teacher_model": self.teacher_model.value,
            "raw_response": response_text
        }

        logger.info(f"Successfully labeled profile. Scores: {personality_scores.to_dict()}")
        return result

    async def batch_label(
        self,
        car_profiles: List[CarProfileInput],
        batch_size: int = 5
    ) -> List[Dict]:
        """
        Label multiple car profiles in batches.

        Args:
            car_profiles: List of car profile inputs
            batch_size: Number of concurrent requests

        Returns:
            List of labeled results
        """
        results = []

        for i in range(0, len(car_profiles), batch_size):
            batch = car_profiles[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(car_profiles) + batch_size - 1) // batch_size}")

            tasks = [self.label_car_profile(profile) for profile in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for profile, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to label profile {profile.customer_id}: {result}")
                else:
                    results.append(result)

        logger.info(f"Batch labeling complete. Successfully labeled {len(results)}/{len(car_profiles)} profiles")
        return results


# Example usage and testing
async def main():
    """Example usage of PsychSteer"""
    # Sample car questionnaire response
    sample_profile = CarProfileInput(
        customer_id="TEST_001",
        responses={
            "Budget Range": "$30,000 - $40,000",
            "Primary Use": "Daily commute and family trips",
            "Family Size": "4 people (2 adults, 2 children)",
            "Key Priorities": "Safety ratings, fuel efficiency, reliability",
            "Aesthetic Preference": "Modern but practical design, not too flashy",
            "Technology Interest": "Interested in backup cameras, blind spot monitoring, but not cutting-edge tech",
            "Environmental Concern": "Somewhat important - considering hybrid options",
            "Maintenance Attitude": "Want a good warranty and low maintenance costs"
        },
        raw_text="I'm looking for a safe, reliable car for my family. Nothing too fancy, but I want good safety features."
    )

    # Initialize PsychSteer
    psychsteer = PsychSteer(
        teacher_model=TeacherModel.GPT4O,
        temperature=0.3
    )

    # Label the profile
    result = await psychsteer.label_car_profile(sample_profile)

    print("\n=== PsychSteer Labeling Result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
