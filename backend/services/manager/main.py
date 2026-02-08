"""
Manager Service - Training Job Orchestration API

Production-ready training orchestration for the Car-Psycho psychometric profiling
platform. Manages training jobs, model registry, real-time metrics streaming, and
lifecycle of personality prediction models trained on car preference data.

Endpoints:
  POST   /jobs                    - Create a new training job
  GET    /jobs                    - List all training jobs
  GET    /jobs/{job_id}           - Get full job details
  POST   /jobs/{job_id}/stop      - Stop a running job
  DELETE /jobs/{job_id}           - Delete a job record
  GET    /models                  - List all saved models
  GET    /models/{model_id}       - Get model details
  POST   /models/{model_id}/load  - Load a model for inference
  DELETE /models/{model_id}       - Delete a model and files
  WS     /ws/training/{job_id}    - Stream real-time training metrics
  GET    /health                  - Detailed health check
  GET    /config/defaults         - Default training configuration
"""

import asyncio
import csv
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import httpx
import redis
import torch
import torch.nn as nn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# ML imports -- the backend/ tree is mounted at /app/backend inside Docker
# ---------------------------------------------------------------------------
sys.path.insert(0, "/app/backend")
try:
    from ml.losses.ordinal_regression import (
        CombinedOrdinalMSELoss,
        OrdinalRegressionLoss,
    )

    _ML_IMPORTS_OK = True
except ImportError:
    _ML_IMPORTS_OK = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("manager")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("/app/data")
MODELS_DIR = Path("/app/models")
SAVED_MODELS_DIR = MODELS_DIR / "saved"
TRAIT_NAMES = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
SERVICE_VERSION = "1.0.0"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

# ---------------------------------------------------------------------------
# In-process state for running jobs
# ---------------------------------------------------------------------------
_active_tasks: Dict[str, asyncio.Task] = {}
_stop_events: Dict[str, asyncio.Event] = {}


# ===================================================================== #
#                         Pydantic Schemas                               #
# ===================================================================== #


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingConfig(BaseModel):
    model_base: str = "personality-mlp"
    dataset_id: Optional[str] = None
    ollama_model: str = Field(default="llama3.2:1b", description="Ollama model for embeddings")
    epochs: int = Field(default=10, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-3, gt=0, le=1.0)
    lm_loss_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    personality_loss_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    ordinal_loss_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    mse_loss_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    car_domain_boost: float = Field(default=1.2, ge=1.0, le=3.0)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    max_seq_length: int = Field(default=512, ge=64)
    warmup_steps: int = Field(default=100, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    save_steps: int = Field(default=50, ge=1)
    eval_steps: int = Field(default=50, ge=1)
    hidden_size: int = Field(default=2048, ge=32)
    intermediate_size: int = Field(default=512, ge=16)
    num_synthetic_samples: int = Field(default=1000, ge=10, le=100000)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)


class CreateJobRequest(BaseModel):
    job_name: str = Field(..., min_length=1, max_length=255)
    config: TrainingConfig = Field(default_factory=TrainingConfig)


class JobSummary(BaseModel):
    id: str
    job_name: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class JobDetail(BaseModel):
    id: str
    job_name: str
    model_id: Optional[str] = None
    config: Dict[str, Any]
    status: str
    progress: float
    current_epoch: int
    total_epochs: Optional[int] = None
    loss_history: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str
    updated_at: str


class ModelSummary(BaseModel):
    id: str
    name: str
    version: str
    model_type: str
    base_model: Optional[str] = None
    description: Optional[str] = None
    storage_path: str
    config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    status: str
    is_default: bool
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: Dict[str, Any]
    redis: Dict[str, Any]
    gpu: Dict[str, Any]
    active_jobs: int


# ===================================================================== #
#                       Database Helpers                                  #
# ===================================================================== #


def _get_db_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


def get_db_connection():
    return psycopg2.connect(_get_db_url())


@contextmanager
def db_cursor(commit: bool = False):
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def _row_to_dict(row) -> dict:
    """Convert a RealDictRow to a plain dict with serialisable values."""
    if row is None:
        return {}
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif isinstance(v, uuid.UUID):
            d[k] = str(v)
    return d


# ===================================================================== #
#                          Redis Helpers                                  #
# ===================================================================== #


def _get_redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://redis:6379")


def get_redis_client() -> redis.Redis:
    return redis.from_url(_get_redis_url(), decode_responses=True)


def publish_metrics(job_id: str, metrics: dict):
    """Publish a metrics payload to the Redis channel for a job."""
    try:
        r = get_redis_client()
        r.publish(f"training:{job_id}:metrics", json.dumps(metrics, default=str))
    except Exception as exc:
        logger.warning("Redis publish failed for job %s: %s", job_id, exc)


# ===================================================================== #
#                 Inline Loss Function Fallback                          #
# ===================================================================== #

if not _ML_IMPORTS_OK:
    logger.warning(
        "Could not import ML losses from backend.ml -- using inline fallback."
    )

    class OrdinalRegressionLoss(nn.Module):
        def __init__(self, num_thresholds: int = 10, temperature: float = 1.0,
                     reduction: str = "mean"):
            super().__init__()
            self.num_thresholds = num_thresholds
            self.temperature = temperature
            self.reduction = reduction
            self.register_buffer(
                "thresholds",
                torch.linspace(0, 1, num_thresholds + 2)[1:-1],
            )

        def forward(self, predictions, targets, trait_weights=None):
            scores = targets.unsqueeze(-1)
            thresholds = self.thresholds.unsqueeze(0).unsqueeze(0)
            ordinal_targets = (scores > thresholds).float()
            preds = predictions.unsqueeze(-1)
            ordinal_preds = torch.sigmoid((preds - thresholds) / self.temperature)
            loss = nn.functional.binary_cross_entropy(
                ordinal_preds, ordinal_targets, reduction="none"
            ).mean(dim=-1)
            if trait_weights is not None:
                loss = loss * trait_weights.unsqueeze(0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss

    class CombinedOrdinalMSELoss(nn.Module):
        def __init__(self, ordinal_weight: float = 0.7, mse_weight: float = 0.3,
                     num_thresholds: int = 10):
            super().__init__()
            self.ordinal_weight = ordinal_weight
            self.mse_weight = mse_weight
            self.ordinal_loss = OrdinalRegressionLoss(num_thresholds=num_thresholds)
            self.mse_loss = nn.MSELoss()

        def forward(self, predictions, targets):
            ord_val = self.ordinal_loss(predictions, targets)
            mse_val = self.mse_loss(predictions, targets)
            total = self.ordinal_weight * ord_val + self.mse_weight * mse_val
            return total, {
                "ordinal_loss": ord_val.item(),
                "mse_loss": mse_val.item(),
                "total_loss": total.item(),
            }


# ===================================================================== #
#               PersonalityPredictionHead (Inline)                       #
# ===================================================================== #


class PersonalityPredictionHead(nn.Module):
    """
    3-layer MLP that maps a feature vector to Big Five personality trait
    scores in [0, 1].  Architecture mirrors the one used by
    OrdinalPsychTrainer so that saved weights are compatible.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 128,
        num_traits: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, intermediate_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size // 2, num_traits),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


# ===================================================================== #
#                      Training Data Loader                              #
# ===================================================================== #


def _load_csv_dataset(path: Path) -> Optional[List[dict]]:
    """Load a CSV dataset and return records as list of dicts."""
    logger.info("Loading CSV dataset from %s", path)
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(dict(row))
    return records if records else None


def _load_jsonl_dataset(path: Path) -> Optional[List[dict]]:
    """Load a JSONL dataset and return records as list of dicts."""
    logger.info("Loading JSONL dataset from %s", path)
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records if records else None


def _load_json_dataset(path: Path) -> Optional[List[dict]]:
    """Load a JSON dataset (array of objects) and return records."""
    logger.info("Loading JSON dataset from %s", path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data if data else None
    return [data] if isinstance(data, dict) else None


def _find_and_load_dataset(config: TrainingConfig) -> Optional[List[dict]]:
    """
    Search /app/data/ for dataset files (CSV, JSONL, JSON) and load the first one found.
    Prioritises exact dataset_id match, then walks all subdirectories.
    """
    candidates: List[Path] = []

    # 1. Exact match by dataset_id
    if config.dataset_id:
        for ext in (".csv", ".jsonl", ".json"):
            candidates.append(DATA_DIR / f"{config.dataset_id}{ext}")
            candidates.append(DATA_DIR / config.dataset_id / f"train{ext}")
            candidates.append(DATA_DIR / config.dataset_id / f"data{ext}")
        # Also try subdirectories matching dataset_id name
        dsdir = DATA_DIR / config.dataset_id
        if dsdir.is_dir():
            for ext in (".csv", ".jsonl", ".json"):
                candidates.extend(sorted(dsdir.glob(f"*{ext}")))

    # 2. Glob all supported files recursively under /app/data/
    for ext in ("*.csv", "*.jsonl", "*.json"):
        candidates.extend(sorted(DATA_DIR.rglob(ext)))

    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    for path in unique:
        if not path.exists() or path.stat().st_size == 0:
            continue
        ext = path.suffix.lower()
        try:
            if ext == ".csv":
                records = _load_csv_dataset(path)
            elif ext == ".jsonl":
                records = _load_jsonl_dataset(path)
            elif ext == ".json":
                records = _load_json_dataset(path)
            else:
                continue
            if records:
                logger.info("Found %d records in %s", len(records), path)
                return records
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            continue
    return None


def _extract_ocean_scores(rec: dict) -> Optional[List[float]]:
    """
    Extract OCEAN personality scores from a single record.

    Supports multiple formats:
    - Direct columns: rec["openness"], rec["conscientiousness"], etc.
    - Nested dict: rec["personality_scores"]["openness"], etc.
    - Nested JSON string: rec["personality_labels"] = '{"openness": 0.7, ...}'
    - List/tuple: rec["scores"] = [0.7, 0.8, ...]
    """
    # Build case-insensitive key map from the record
    rec_lower = {k.strip().lower(): v for k, v in rec.items()}

    # Try direct trait columns (common in CSV: "openness", "Openness", "OPENNESS")
    direct = []
    for t in TRAIT_NAMES:
        val = rec.get(t) or rec_lower.get(t)
        if val is not None:
            try:
                direct.append(float(val))
            except (ValueError, TypeError):
                break
    if len(direct) == 5:
        return direct

    # Try prefixed columns (big5_openness, ocean_openness)
    prefixed = []
    for t in TRAIT_NAMES:
        val = rec_lower.get(f"big5_{t}") or rec_lower.get(f"ocean_{t}")
        if val is not None:
            try:
                prefixed.append(float(val))
            except (ValueError, TypeError):
                break
    if len(prefixed) == 5:
        return prefixed

    # Try nested dict keys
    for key in ("personality_scores", "scores", "personality_labels", "labels", "ocean"):
        ps = rec.get(key)
        if ps is None:
            continue
        # If it's a JSON string, parse it
        if isinstance(ps, str):
            try:
                ps = json.loads(ps)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(ps, dict):
            row = []
            for t in TRAIT_NAMES:
                val = ps.get(t, ps.get(t[0]))  # try full name or first letter
                if val is not None:
                    try:
                        row.append(float(val))
                    except (ValueError, TypeError):
                        break
            if len(row) == 5:
                return row
        elif isinstance(ps, (list, tuple)) and len(ps) >= 5:
            try:
                return [float(x) for x in ps[:5]]
            except (ValueError, TypeError):
                continue
    # -- Pandora / trait-level format ----------------------------------------
    # e.g. {"trait": "openness", "level": "low"} -> assign a numeric score
    # for the named trait, defaulting the rest to 0.5
    trait_val = rec_lower.get("trait")
    level_val = rec_lower.get("level")
    if trait_val and level_val:
        trait_str = str(trait_val).strip().lower()
        level_str = str(level_val).strip().lower()
        level_map = {
            "very low": 0.1, "low": 0.25, "below average": 0.35,
            "average": 0.5, "above average": 0.65,
            "high": 0.75, "very high": 0.9,
        }
        if trait_str in TRAIT_NAMES and level_str in level_map:
            scores = [0.5] * 5
            idx = TRAIT_NAMES.index(trait_str)
            scores[idx] = level_map[level_str]
            return scores

    return None


# Common text column names in personality datasets
TEXT_COLUMN_NAMES = {
    "text", "content", "message", "comment", "response", "post",
    "description", "essay", "writing", "answer", "body", "input",
    "status", "tweet", "review", "title", "question", "prompt",
    "sentence", "utterance", "reply", "chat",
    # Pandora / training data columns
    "train_input", "train_output", "narrative", "literal",
    "instruction", "train_instruction",
}


def _extract_text(rec: dict) -> Optional[str]:
    """Extract text content from a record, checking common column names."""
    rec_lower = {k.strip().lower(): v for k, v in rec.items()}
    for name in TEXT_COLUMN_NAMES:
        val = rec_lower.get(name)
        if val and isinstance(val, str) and len(val.strip()) > 5:
            return val.strip()
    # Fallback: use the longest string value in the record
    longest = ""
    for v in rec.values():
        if isinstance(v, str) and len(v) > len(longest):
            longest = v
    return longest.strip() if len(longest) > 5 else None


async def get_ollama_embeddings(
    texts: List[str],
    model: str,
    batch_size: int = 32,
) -> List[List[float]]:
    """Batch-fetch embeddings from Ollama /api/embed endpoint."""
    all_embeddings: List[List[float]] = []
    total = len(texts)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=30.0)
    ) as client:
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = await client.post(
                    f"{OLLAMA_HOST}/api/embed",
                    json={"model": model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                all_embeddings.extend(data["embeddings"])
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Ollama HTTP %d at batch %d/%d: %s",
                    exc.response.status_code, i, total, exc.response.text[:200],
                )
                raise RuntimeError(
                    f"Ollama embedding failed (HTTP {exc.response.status_code}). "
                    f"Is '{model}' pulled? Run: ollama pull {model}"
                ) from exc
            except Exception as exc:
                logger.error("Ollama embed error at batch %d/%d: %s", i, total, exc)
                raise RuntimeError(f"Ollama connection failed: {exc}") from exc

            logger.info(
                "Embeddings: %d/%d samples", min(i + batch_size, total), total
            )
            await asyncio.sleep(0)

    return all_embeddings


async def load_training_data_async(
    config: TrainingConfig,
    job_id: str,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Return (inputs, targets, embedding_dim) using Ollama embeddings.

    *inputs*  -- shape [N, embedding_dim] (Ollama embeddings)
    *targets* -- shape [N, 5]             (OCEAN scores in [0, 1])

    Loads CSV/JSONL/JSON data, extracts text + OCEAN scores, then calls
    Ollama to generate embeddings for each text sample.
    """
    records = await asyncio.get_event_loop().run_in_executor(
        None, _find_and_load_dataset, config
    )

    texts: List[str] = []
    targets_list: List[List[float]] = []

    if records:
        for rec in records:
            scores = _extract_ocean_scores(rec)
            text = _extract_text(rec)
            if scores and text:
                texts.append(text)
                targets_list.append(scores)
        if not texts:
            logger.warning(
                "Found %d records but none had BOTH text AND OCEAN scores. "
                "Need a text column (%s) plus OCEAN columns (%s).",
                len(records),
                ", ".join(sorted(TEXT_COLUMN_NAMES)[:6]),
                ", ".join(TRAIT_NAMES),
            )

    if texts and targets_list:
        n = len(texts)
        logger.info("Loaded %d records with text + OCEAN scores.", n)

        # Publish status so the frontend sees embedding progress
        publish_metrics(job_id, {
            "job_id": job_id,
            "type": "status",
            "message": f"Generating embeddings for {n} samples via {config.ollama_model}...",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        embeddings = await get_ollama_embeddings(
            texts, config.ollama_model, batch_size=config.batch_size
        )

        embedding_dim = len(embeddings[0])
        logger.info(
            "Ollama embeddings ready: %d samples, dim=%d (model=%s)",
            n, embedding_dim, config.ollama_model,
        )

        inputs = torch.tensor(embeddings, dtype=torch.float32)
        targets = torch.tensor(targets_list, dtype=torch.float32).clamp(0, 1)
        return inputs, targets, embedding_dim

    # -- Fallback: synthetic random data (no real embeddings) ----------------
    n = config.num_synthetic_samples
    logger.info("No suitable data found -- generating %d synthetic samples.", n)
    inputs = torch.randn(n, config.hidden_size)
    targets = torch.rand(n, 5)
    return inputs, targets, config.hidden_size


# ===================================================================== #
#                        Training Engine                                  #
# ===================================================================== #


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


async def run_training_job(job_id: str, job_name: str, config: TrainingConfig):
    """
    Execute a full training run as an asyncio background coroutine.

    Lifecycle:
      1. Mark job as *running* in Postgres.
      2. Build model + loss + optimizer.
      3. Run epoch/step loop with real gradient updates.
      4. Publish per-step metrics to Redis.
      5. Persist epoch-level summaries to Postgres (loss_history, progress).
      6. Save model checkpoint per epoch + final model.
      7. Register final model in the *models* table.
      8. Mark job as *completed* (or *failed* / *stopped*).
    """
    stop_event = _stop_events.get(job_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Job %s starting on device=%s", job_id, device)

    # -- Mark RUNNING -----------------------------------------------------
    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """UPDATE training_jobs
                      SET status = 'running',
                          started_at = NOW(),
                          total_epochs = %s,
                          updated_at = NOW()
                    WHERE id = %s""",
                (config.epochs, job_id),
            )
    except Exception as exc:
        logger.error("DB error marking job running: %s", exc)

    try:
        # -- Data (Ollama embeddings) -------------------------------------
        inputs, targets, embedding_dim = await load_training_data_async(
            config, job_id
        )
        inputs = inputs.to(device)
        targets = targets.to(device)
        num_samples = inputs.size(0)
        logger.info(
            "Training data: %d samples, embedding dim=%d (model=%s)",
            num_samples, embedding_dim, config.ollama_model,
        )

        # -- Model (auto-size input to match Ollama embeddings) -----------
        actual_hidden = embedding_dim
        if actual_hidden != config.hidden_size:
            logger.info(
                "Auto-adjusting hidden_size %d -> %d to match Ollama embeddings",
                config.hidden_size, actual_hidden,
            )
        model = PersonalityPredictionHead(
            hidden_size=actual_hidden,
            intermediate_size=config.intermediate_size,
            num_traits=5,
            dropout=config.dropout,
        ).to(device)

        # -- Loss ---------------------------------------------------------
        loss_fn = CombinedOrdinalMSELoss(
            ordinal_weight=config.ordinal_loss_weight,
            mse_weight=config.mse_loss_weight,
        ).to(device)

        # -- Optimizer + scheduler ----------------------------------------
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        batch_size = config.batch_size
        steps_per_epoch = max(1, math.ceil(num_samples / batch_size))
        total_steps = steps_per_epoch * config.epochs
        grad_accum = config.gradient_accumulation_steps

        loss_history: List[Dict[str, Any]] = []
        global_step = 0

        # -- Checkpoint directory -----------------------------------------
        job_ckpt_dir = SAVED_MODELS_DIR / job_name
        job_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # -- Training loop ------------------------------------------------
        for epoch in range(1, config.epochs + 1):
            if stop_event and stop_event.is_set():
                logger.info("Job %s: stop signal received before epoch %d", job_id, epoch)
                break

            model.train()
            epoch_losses: List[Dict[str, float]] = []
            epoch_preds: List[torch.Tensor] = []
            epoch_targets: List[torch.Tensor] = []

            # Shuffle indices each epoch
            perm = torch.randperm(num_samples, device=device)
            optimizer.zero_grad()

            for step_idx in range(steps_per_epoch):
                if stop_event and stop_event.is_set():
                    break

                start = step_idx * batch_size
                end = min(start + batch_size, num_samples)
                idx = perm[start:end]
                batch_x = inputs[idx]
                batch_y = targets[idx]

                preds = model(batch_x)
                loss, loss_dict = loss_fn(preds, batch_y)
                scaled_loss = loss / grad_accum
                scaled_loss.backward()

                # Gradient accumulation
                if (step_idx + 1) % grad_accum == 0 or (step_idx + 1) == steps_per_epoch:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                global_step += 1

                # Adjust LR
                new_lr = _cosine_lr(global_step, total_steps, config.warmup_steps, config.learning_rate)
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr

                epoch_losses.append(loss_dict)
                epoch_preds.append(preds.detach())
                epoch_targets.append(batch_y.detach())

                # -- Per-step Redis metrics -------------------------------
                step_metrics = {
                    "job_id": job_id,
                    "type": "step",
                    "epoch": epoch,
                    "step": global_step,
                    "step_in_epoch": step_idx + 1,
                    "steps_per_epoch": steps_per_epoch,
                    "learning_rate": new_lr,
                    "total_loss": loss_dict["total_loss"],
                    "ordinal_loss": loss_dict["ordinal_loss"],
                    "mse_loss": loss_dict["mse_loss"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                publish_metrics(job_id, step_metrics)

                # Yield control so the event loop stays responsive
                if global_step % 5 == 0:
                    await asyncio.sleep(0)

            # -- Epoch-level aggregation ----------------------------------
            if stop_event and stop_event.is_set():
                break

            avg_total = sum(d["total_loss"] for d in epoch_losses) / len(epoch_losses)
            avg_ordinal = sum(d["ordinal_loss"] for d in epoch_losses) / len(epoch_losses)
            avg_mse = sum(d["mse_loss"] for d in epoch_losses) / len(epoch_losses)

            # Per-trait MAE
            all_preds = torch.cat(epoch_preds, dim=0)
            all_targets = torch.cat(epoch_targets, dim=0)
            trait_mae = (all_preds - all_targets).abs().mean(dim=0)
            trait_mae_dict = {
                TRAIT_NAMES[i]: round(trait_mae[i].item(), 6) for i in range(5)
            }

            epoch_summary = {
                "epoch": epoch,
                "total_loss": round(avg_total, 6),
                "ordinal_loss": round(avg_ordinal, 6),
                "mse_loss": round(avg_mse, 6),
                "personality_loss": round(avg_total, 6),
                "mean_trait_mae": round(trait_mae.mean().item(), 6),
                "per_trait_mae": trait_mae_dict,
                "learning_rate": new_lr,
                "global_step": global_step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            loss_history.append(epoch_summary)

            # Publish epoch summary via Redis
            epoch_metrics = {**epoch_summary, "job_id": job_id, "type": "epoch"}
            publish_metrics(job_id, epoch_metrics)

            progress = round(epoch / config.epochs, 4)

            # -- Persist to Postgres --------------------------------------
            try:
                with db_cursor(commit=True) as cur:
                    cur.execute(
                        """UPDATE training_jobs
                              SET current_epoch = %s,
                                  progress = %s,
                                  loss_history = %s,
                                  metrics = %s,
                                  updated_at = NOW()
                            WHERE id = %s""",
                        (
                            epoch,
                            progress,
                            json.dumps(loss_history),
                            json.dumps(epoch_summary),
                            job_id,
                        ),
                    )
            except Exception as exc:
                logger.warning("DB update failed at epoch %d: %s", epoch, exc)

            # -- Save checkpoint ------------------------------------------
            ckpt_path = job_ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.model_dump(),
                    "loss_history": loss_history,
                    "embedding_dim": embedding_dim,
                    "ollama_model": config.ollama_model,
                },
                str(ckpt_path),
            )
            logger.info(
                "Epoch %d/%d -- loss=%.5f  mae=%.5f  saved=%s",
                epoch, config.epochs, avg_total,
                trait_mae.mean().item(), ckpt_path.name,
            )

        # -- Determine final status ---------------------------------------
        was_stopped = stop_event and stop_event.is_set()
        final_status = "stopped" if was_stopped else "completed"

        # -- Save final model ---------------------------------------------
        final_path = job_ckpt_dir / "model_final.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config.model_dump(),
                "loss_history": loss_history,
                "trait_names": TRAIT_NAMES,
                "ollama_model": config.ollama_model,
                "embedding_dim": embedding_dim,
            },
            str(final_path),
        )

        # Also save config as JSON for easier inspection
        with open(job_ckpt_dir / "config.json", "w") as fh:
            json.dump(config.model_dump(), fh, indent=2)

        # -- Register model in DB -----------------------------------------
        model_id = None
        if final_status == "completed" and loss_history:
            try:
                model_version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                final_metrics = loss_history[-1] if loss_history else {}
                with db_cursor(commit=True) as cur:
                    cur.execute(
                        """INSERT INTO models
                               (name, version, model_type, base_model,
                                description, storage_path, config, metrics, status)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active')
                           RETURNING id""",
                        (
                            job_name,
                            model_version,
                            "personality-prediction-head",
                            config.model_base,
                            f"Trained for {config.epochs} epochs on {config.ollama_model}. "
                            f"Final loss={final_metrics.get('total_loss', 'N/A')}",
                            str(job_ckpt_dir),
                            json.dumps(config.model_dump()),
                            json.dumps(final_metrics),
                        ),
                    )
                    row = cur.fetchone()
                    model_id = str(row["id"]) if row else None
                    logger.info("Registered model %s (id=%s)", job_name, model_id)
            except Exception as exc:
                logger.error("Failed to register model: %s", exc)

        # -- Final job update ---------------------------------------------
        try:
            with db_cursor(commit=True) as cur:
                cur.execute(
                    """UPDATE training_jobs
                          SET status = %s,
                              progress = %s,
                              loss_history = %s,
                              metrics = %s,
                              model_id = %s,
                              completed_at = NOW(),
                              updated_at = NOW()
                        WHERE id = %s""",
                    (
                        final_status,
                        1.0 if final_status == "completed" else progress,
                        json.dumps(loss_history),
                        json.dumps(loss_history[-1]) if loss_history else None,
                        model_id,
                        job_id,
                    ),
                )
        except Exception as exc:
            logger.error("DB error on final update: %s", exc)

        # Publish completion event
        publish_metrics(job_id, {
            "job_id": job_id,
            "type": "complete",
            "status": final_status,
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Job %s finished with status=%s", job_id, final_status)

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Job %s failed: %s\n%s", job_id, exc, tb)
        error_msg = f"{exc}\n{tb}"
        try:
            with db_cursor(commit=True) as cur:
                cur.execute(
                    """UPDATE training_jobs
                          SET status = 'failed',
                              error_message = %s,
                              completed_at = NOW(),
                              updated_at = NOW()
                        WHERE id = %s""",
                    (error_msg[:4000], job_id),
                )
        except Exception as db_exc:
            logger.error("DB error recording failure: %s", db_exc)
        publish_metrics(job_id, {
            "job_id": job_id,
            "type": "error",
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    finally:
        _active_tasks.pop(job_id, None)
        _stop_events.pop(job_id, None)


# ===================================================================== #
#                          FastAPI App                                    #
# ===================================================================== #

app = FastAPI(
    title="Car-Psycho Manager Service",
    description="Training orchestration API for psychometric profiling models",
    version=SERVICE_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
#                         Root                                         #
# ------------------------------------------------------------------ #

@app.get("/")
async def root():
    return {"service": "manager", "status": "running", "version": SERVICE_VERSION}


# ------------------------------------------------------------------ #
#                      Job Endpoints                                   #
# ------------------------------------------------------------------ #

@app.post("/jobs", status_code=201)
async def create_job(request: CreateJobRequest):
    """Create a new training job and launch it as a background task."""
    config = request.config
    job_id: Optional[str] = None

    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO training_jobs
                       (job_name, config, status, progress, current_epoch, total_epochs)
                   VALUES (%s, %s, 'pending', 0.0, 0, %s)
                   RETURNING id""",
                (request.job_name, json.dumps(config.model_dump()), config.epochs),
            )
            row = cur.fetchone()
            job_id = str(row["id"])
    except Exception as exc:
        logger.error("Failed to insert job: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    # Prepare stop event and launch background task
    stop_event = asyncio.Event()
    _stop_events[job_id] = stop_event
    task = asyncio.create_task(
        run_training_job(job_id, request.job_name, config)
    )
    _active_tasks[job_id] = task

    return {
        "id": job_id,
        "job_name": request.job_name,
        "status": "pending",
        "message": "Training job created and queued.",
    }


@app.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List training jobs with optional status filter."""
    try:
        with db_cursor() as cur:
            if status:
                cur.execute(
                    """SELECT id, job_name, status, progress, current_epoch,
                              total_epochs, metrics, created_at, updated_at
                         FROM training_jobs
                        WHERE status = %s
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s""",
                    (status, limit, offset),
                )
            else:
                cur.execute(
                    """SELECT id, job_name, status, progress, current_epoch,
                              total_epochs, metrics, created_at, updated_at
                         FROM training_jobs
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s""",
                    (limit, offset),
                )
            rows = cur.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    jobs = [_row_to_dict(r) for r in rows]
    return {"jobs": jobs, "count": len(jobs), "limit": limit, "offset": offset}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get full details for a single training job."""
    try:
        with db_cursor() as cur:
            cur.execute("SELECT * FROM training_jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    data = _row_to_dict(row)
    data["is_active"] = job_id in _active_tasks
    return data


@app.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """Request graceful stop of a running training job."""
    event = _stop_events.get(job_id)
    if not event:
        # Check if job exists at all
        try:
            with db_cursor() as cur:
                cur.execute(
                    "SELECT status FROM training_jobs WHERE id = %s", (job_id,)
                )
                row = cur.fetchone()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Database error: {exc}")

        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        if row["status"] in ("completed", "failed", "stopped"):
            return {"message": f"Job already {row['status']}.", "status": row["status"]}
        raise HTTPException(status_code=409, detail="Job is not actively running in this process.")

    event.set()
    logger.info("Stop signal sent to job %s", job_id)
    return {"message": "Stop signal sent. Job will finish the current step and halt.", "job_id": job_id}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a training job record. Running jobs must be stopped first."""
    if job_id in _active_tasks:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a running job. Stop it first.",
        )

    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                "DELETE FROM training_jobs WHERE id = %s RETURNING id", (job_id,)
            )
            row = cur.fetchone()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"message": "Job deleted.", "id": job_id}


# ------------------------------------------------------------------ #
#                      Model Endpoints                                 #
# ------------------------------------------------------------------ #

@app.get("/models")
async def list_models(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List all registered models."""
    try:
        with db_cursor() as cur:
            if status:
                cur.execute(
                    """SELECT * FROM models
                        WHERE status = %s
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s""",
                    (status, limit, offset),
                )
            else:
                cur.execute(
                    """SELECT * FROM models
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s""",
                    (limit, offset),
                )
            rows = cur.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    return {"models": [_row_to_dict(r) for r in rows], "count": len(rows)}


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get details for a single model."""
    try:
        with db_cursor() as cur:
            cur.execute("SELECT * FROM models WHERE id = %s", (model_id,))
            row = cur.fetchone()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not row:
        raise HTTPException(status_code=404, detail="Model not found")

    data = _row_to_dict(row)

    # Check if files exist on disk
    storage = Path(data.get("storage_path", ""))
    data["files_exist"] = storage.exists()
    if storage.exists():
        files = [f.name for f in storage.iterdir() if f.is_file()]
        data["files"] = files

    return data


@app.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """
    Load a saved model into memory for inference readiness verification.
    Returns model metadata and confirms the checkpoint is loadable.
    """
    try:
        with db_cursor() as cur:
            cur.execute("SELECT * FROM models WHERE id = %s", (model_id,))
            row = cur.fetchone()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not row:
        raise HTTPException(status_code=404, detail="Model not found")

    data = _row_to_dict(row)
    storage = Path(data["storage_path"])
    final_ckpt = storage / "model_final.pt"

    if not final_ckpt.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model checkpoint not found at {final_ckpt}",
        )

    try:
        ckpt = torch.load(str(final_ckpt), map_location="cpu", weights_only=False)
        model_config = ckpt.get("config", {})
        head = PersonalityPredictionHead(
            hidden_size=model_config.get("hidden_size", 256),
            intermediate_size=model_config.get("intermediate_size", 128),
        )
        head.load_state_dict(ckpt["model_state_dict"])
        head.eval()
        num_params = sum(p.numel() for p in head.parameters())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {exc}",
        )

    return {
        "message": "Model loaded successfully.",
        "model_id": model_id,
        "name": data["name"],
        "version": data["version"],
        "num_parameters": num_params,
        "config": model_config,
        "trait_names": ckpt.get("trait_names", TRAIT_NAMES),
    }


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model record and its files from disk."""
    try:
        with db_cursor() as cur:
            cur.execute("SELECT * FROM models WHERE id = %s", (model_id,))
            row = cur.fetchone()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not row:
        raise HTTPException(status_code=404, detail="Model not found")

    data = _row_to_dict(row)
    storage = Path(data["storage_path"])

    # Remove files
    if storage.exists() and storage.is_dir():
        try:
            shutil.rmtree(str(storage))
            logger.info("Deleted model files at %s", storage)
        except Exception as exc:
            logger.error("Failed to delete model files: %s", exc)

    # Remove DB record
    try:
        with db_cursor(commit=True) as cur:
            cur.execute("DELETE FROM models WHERE id = %s", (model_id,))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    return {"message": "Model deleted.", "id": model_id, "files_removed": not storage.exists()}


# ------------------------------------------------------------------ #
#                       WebSocket Endpoint                             #
# ------------------------------------------------------------------ #

@app.websocket("/ws/training/{job_id}")
async def ws_training(websocket: WebSocket, job_id: str):
    """
    Stream real-time training metrics for a job over WebSocket.

    The server subscribes to the Redis pub/sub channel
    ``training:{job_id}:metrics`` and forwards every message to the
    connected client.  The connection stays open until the client
    disconnects or the training completes/fails.
    """
    await websocket.accept()
    logger.info("WebSocket connected for job %s", job_id)

    r: Optional[redis.Redis] = None
    pubsub = None

    try:
        r = get_redis_client()
        pubsub = r.pubsub()
        channel = f"training:{job_id}:metrics"
        pubsub.subscribe(channel)

        while True:
            # Non-blocking check for new messages
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
            if message and message["type"] == "message":
                await websocket.send_text(message["data"])

                # Check if this was a terminal event
                try:
                    payload = json.loads(message["data"])
                    if payload.get("type") in ("complete", "error"):
                        # Send one last message then close
                        await asyncio.sleep(0.1)
                        break
                except (json.JSONDecodeError, KeyError):
                    pass

            # Also check for client disconnect
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                # Client can send "ping" to keep alive or "stop" to request stop
                if data.strip().lower() == "stop":
                    event = _stop_events.get(job_id)
                    if event:
                        event.set()
                        await websocket.send_text(
                            json.dumps({"type": "ack", "message": "Stop signal sent."})
                        )
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s", job_id)
    except Exception as exc:
        logger.error("WebSocket error for job %s: %s", job_id, exc)
    finally:
        if pubsub:
            try:
                pubsub.unsubscribe()
                pubsub.close()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass


# ------------------------------------------------------------------ #
#                        Health & Config                                #
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    """Detailed health check covering database, Redis, and GPU status."""
    # -- Database ---------------------------------------------------------
    db_status: Dict[str, Any] = {"connected": False}
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        db_status = {"connected": True, "url_configured": True}
    except Exception as exc:
        db_status = {"connected": False, "error": str(exc)}

    # -- Redis ------------------------------------------------------------
    redis_status: Dict[str, Any] = {"connected": False}
    try:
        r = get_redis_client()
        r.ping()
        info = r.info("server")
        redis_status = {
            "connected": True,
            "version": info.get("redis_version", "unknown"),
        }
    except Exception as exc:
        redis_status = {"connected": False, "error": str(exc)}

    # -- GPU --------------------------------------------------------------
    gpu_status: Dict[str, Any] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        gpu_status["device_count"] = torch.cuda.device_count()
        gpu_status["device_name"] = torch.cuda.get_device_name(0)
        gpu_status["memory_allocated_mb"] = round(
            torch.cuda.memory_allocated(0) / 1024 / 1024, 2
        )
        gpu_status["memory_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1024 / 1024, 2
        )

    # -- Ollama -----------------------------------------------------------
    ollama_status: Dict[str, Any] = {"connected": False}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags")
            if resp.status_code == 200:
                tags = resp.json()
                model_names = [m["name"] for m in tags.get("models", [])]
                ollama_status = {
                    "connected": True,
                    "host": OLLAMA_HOST,
                    "models": model_names,
                }
    except Exception as exc:
        ollama_status = {"connected": False, "host": OLLAMA_HOST, "error": str(exc)}

    overall = "healthy" if db_status["connected"] and redis_status["connected"] else "degraded"

    return {
        "status": overall,
        "service": "manager",
        "version": SERVICE_VERSION,
        "database": db_status,
        "redis": redis_status,
        "gpu": gpu_status,
        "ollama": ollama_status,
        "active_jobs": len(_active_tasks),
        "ml_imports": _ML_IMPORTS_OK,
        "data_dir_exists": DATA_DIR.exists(),
        "models_dir_exists": MODELS_DIR.exists(),
    }


@app.get("/config/defaults")
async def config_defaults():
    """Return the default training configuration with descriptions."""
    defaults = TrainingConfig()
    return {
        "defaults": defaults.model_dump(),
        "descriptions": {
            "model_base": "Base model identifier or architecture name.",
            "dataset_id": "ID or filename stem of the dataset in /app/data/.",
            "ollama_model": "Ollama model used for text embeddings (e.g. llama3.2:1b).",
            "epochs": "Number of full passes over the training data.",
            "batch_size": "Samples per forward pass.",
            "learning_rate": "Peak learning rate for AdamW optimizer.",
            "lm_loss_weight": "Weight for language-modeling loss component.",
            "personality_loss_weight": "Weight for personality prediction loss.",
            "ordinal_loss_weight": "Weight for ordinal regression within personality loss.",
            "mse_loss_weight": "Weight for MSE within personality loss.",
            "car_domain_boost": "Multiplicative boost for car-domain samples.",
            "gradient_accumulation_steps": "Steps to accumulate gradients before optimizer step.",
            "max_seq_length": "Maximum input sequence length (tokens).",
            "warmup_steps": "Linear warmup steps before cosine decay.",
            "weight_decay": "L2 regularization coefficient.",
            "save_steps": "Save a checkpoint every N steps.",
            "eval_steps": "Run evaluation every N steps.",
            "hidden_size": "Feature dimensionality for the prediction head input.",
            "intermediate_size": "Hidden layer size within the prediction head.",
            "num_synthetic_samples": "Number of synthetic samples if no real data found.",
            "dropout": "Dropout probability in the prediction head.",
        },
        "trait_names": TRAIT_NAMES,
    }


# ------------------------------------------------------------------ #
#                    Startup / Shutdown Events                         #
# ------------------------------------------------------------------ #

@app.on_event("startup")
async def on_startup():
    """Ensure required directories exist on startup."""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Manager service started (version=%s, device=%s, ml_imports=%s)",
        SERVICE_VERSION,
        "cuda" if torch.cuda.is_available() else "cpu",
        _ML_IMPORTS_OK,
    )


@app.on_event("shutdown")
async def on_shutdown():
    """Signal all running jobs to stop and wait briefly for cleanup."""
    logger.info("Shutdown: signalling %d active jobs to stop.", len(_stop_events))
    for event in _stop_events.values():
        event.set()
    # Give tasks a moment to react
    if _active_tasks:
        await asyncio.sleep(2)
    for task in _active_tasks.values():
        if not task.done():
            task.cancel()
    logger.info("Manager service shut down.")
