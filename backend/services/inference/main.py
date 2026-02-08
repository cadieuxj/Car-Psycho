"""
Inference Service - Production Model Serving API
Handles trained model serving, OCEAN personality predictions, and analysis.
"""

import json, logging, os, re, time, uuid
from typing import Any, Dict, List, Optional

import psutil
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIT_NAMES = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
TRAIT_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "openness":          {"high": "Curious, creative, open to new experiences and unconventional ideas.",
                          "low":  "Practical, conventional, prefers routine and familiar experiences."},
    "conscientiousness": {"high": "Organized, dependable, self-disciplined, and goal-oriented.",
                          "low":  "Flexible, spontaneous, may struggle with structure and deadlines."},
    "extraversion":      {"high": "Outgoing, energetic, talkative, and enjoys social interactions.",
                          "low":  "Reserved, introspective, prefers solitude or small-group settings."},
    "agreeableness":     {"high": "Cooperative, trusting, empathetic, and considerate of others.",
                          "low":  "Competitive, skeptical, values self-interest over group harmony."},
    "neuroticism":       {"high": "Emotionally reactive, prone to stress, anxiety, and mood swings.",
                          "low":  "Emotionally stable, calm, resilient under pressure."},
}
POSITIVE_WORDS = frozenset(
    "good great love happy excellent wonderful fantastic amazing enjoy beautiful "
    "best brilliant cheerful delightful excited glad joy kind nice perfect "
    "pleased positive superb thankful thrilled warm".split()
)
NEGATIVE_WORDS = frozenset(
    "bad terrible hate sad awful horrible worst angry annoyed boring cruel "
    "depressed disappointed disgusted dreadful fearful frustrated gloomy harsh "
    "miserable negative painful ugly unhappy worried".split()
)
MODELS_DIR = "/app/models/saved"
_MODEL_COLS = ("id, name, version, model_type, base_model, description, "
               "storage_path, config, metrics, status, is_default, created_at, updated_at")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text for personality prediction")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts for batch prediction")

class OceanScores(BaseModel):
    openness: float = Field(..., ge=0.0, le=1.0)
    conscientiousness: float = Field(..., ge=0.0, le=1.0)
    extraversion: float = Field(..., ge=0.0, le=1.0)
    agreeableness: float = Field(..., ge=0.0, le=1.0)
    neuroticism: float = Field(..., ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    scores: OceanScores
    model_id: str
    model_name: str
    inference_time_ms: float

class BatchPredictResponse(BaseModel):
    predictions: List[OceanScores]
    model_id: str
    model_name: str
    count: int
    total_inference_time_ms: float

class TraitDetail(BaseModel):
    trait: str
    score: float
    label: str
    description: str
    confidence: float

class AnalyzeResponse(BaseModel):
    scores: OceanScores
    traits: List[TraitDetail]
    dominant_trait: str
    model_id: str
    model_name: str
    inference_time_ms: float

class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    model_type: Optional[str] = None
    base_model: Optional[str] = None
    description: Optional[str] = None
    storage_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    is_default: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class ActiveModelResponse(BaseModel):
    loaded: bool
    model: Optional[ModelInfo] = None
    device: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: str
    model_loaded: bool
    loaded_model_name: Optional[str] = None
    device: str
    memory: Dict[str, Any]

# ---------------------------------------------------------------------------
# PersonalityPredictionHead (mirrors training architecture exactly)
# ---------------------------------------------------------------------------
class PersonalityPredictionHead(nn.Module):
    """3-layer MLP: input_size -> intermediate -> 5 traits with Sigmoid."""
    def __init__(self, hidden_size: int, intermediate_size: int = 256,
                 num_traits: int = 5, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(intermediate_size, intermediate_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(intermediate_size // 2, num_traits), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
class _ModelState:
    def __init__(self) -> None:
        self.model: Optional[PersonalityPredictionHead] = None
        self.model_info: Optional[Dict[str, Any]] = None
        self.config: Optional[Dict[str, Any]] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def clear(self) -> None:
        del self.model
        self.model = None
        self.model_info = None
        self.config = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

_state = _ModelState()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def _get_connection():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)

def _db_available() -> bool:
    try:
        _get_connection().close()
        return True
    except Exception:
        return False

def _row_to_info(row: dict) -> ModelInfo:
    return ModelInfo(
        id=str(row["id"]), name=row["name"], version=row["version"],
        model_type=row.get("model_type"), base_model=row.get("base_model"),
        description=row.get("description"), storage_path=row.get("storage_path"),
        config=row.get("config"), metrics=row.get("metrics"),
        status=row.get("status"), is_default=bool(row.get("is_default", False)),
        created_at=row["created_at"].isoformat() if row.get("created_at") else None,
        updated_at=row["updated_at"].isoformat() if row.get("updated_at") else None,
    )

def _query_model(model_id: str) -> dict:
    """Fetch a single model row by UUID or raise 404."""
    try:
        uuid.UUID(model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID format")
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT {_MODEL_COLS} FROM models WHERE id = %s", (model_id,))
        row = cur.fetchone()
        cur.close(); conn.close()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("DB error for model %s: %s", model_id, exc)
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")
    if row is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return row

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(text: str, feature_size: int = 768) -> torch.Tensor:
    """Convert raw text into a fixed-size feature vector (20 NLP features, zero-padded)."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    wc = max(len(words), 1)
    uniq = set(words)
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sc = max(len(sents), 1)
    tlen = max(len(text), 1)
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    fp = sum(1 for w in words if w in {"i", "me", "my", "mine", "myself"})
    hedge = sum(1 for w in words if w in {"maybe", "perhaps", "possibly", "might",
                "could", "seems", "apparently", "somewhat", "likely", "unlikely"})
    punc = sum(1 for c in text if c in ".,;:!?-()\"'")
    bigs = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)] if len(words) > 1 else []
    stops = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in", "on", "at", "to", "for"}

    feats = [
        min(wc / 500, 1.0),                                        # word count
        min(sum(len(w) for w in words) / (wc * 10), 1.0),          # avg word length
        len(uniq) / wc,                                             # type-token ratio
        pos / wc,                                                   # positive ratio
        neg / wc,                                                   # negative ratio
        text.count("?") / tlen,                                     # question marks
        text.count("!") / tlen,                                     # exclamation marks
        text.count(",") / tlen,                                     # comma density
        min(wc / (sc * 30), 1.0),                                   # avg sentence length
        sum(1 for c in text if c.isupper()) / tlen,                 # uppercase ratio
        sum(1 for c in text if c.isdigit()) / tlen,                 # digit ratio
        sum(1 for w in words if len(w) > 6) / wc,                  # long word ratio
        sum(1 for w in words if len(w) <= 3) / wc,                 # short word ratio
        sum(1 for w in words if w in stops) / wc,                   # stopword density
        fp / wc,                                                    # first-person ratio
        punc / tlen,                                                # punctuation density
        len(set(bigs)) / max(len(bigs), 1) if bigs else 0.0,       # unique bigram ratio
        min(text.count("\n") / 20, 1.0),                            # paragraph count
        hedge / wc,                                                 # hedging ratio
        (pos - neg) / wc,                                           # sentiment polarity
    ]
    t = torch.zeros(feature_size)
    t[:len(feats)] = torch.tensor(feats, dtype=torch.float32)
    return t

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Car-Psycho Inference Service",
    description="Production model serving for OCEAN personality predictions",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -- Root / Health ----------------------------------------------------------

@app.get("/")
async def root():
    return {"service": "inference", "status": "running", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health():
    mem = psutil.Process(os.getpid()).memory_info()
    return HealthResponse(
        status="healthy", service="inference", version="1.0.0",
        database="connected" if _db_available() else "unavailable",
        model_loaded=_state.is_loaded,
        loaded_model_name=_state.model_info["name"] if _state.model_info else None,
        device=_state.device,
        memory={
            "rss_mb": round(mem.rss / 1048576, 2),
            "vms_mb": round(mem.vms / 1048576, 2),
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_allocated_mb": (
                round(torch.cuda.memory_allocated() / 1048576, 2)
                if torch.cuda.is_available() else 0
            ),
        },
    )

# -- Model management -------------------------------------------------------

@app.get("/models")
async def list_models():
    """List all registered models from the database."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT {_MODEL_COLS} FROM models ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close(); conn.close()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to query models table: %s", exc)
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")
    models = [_row_to_info(r).model_dump() for r in rows]
    return {"models": models, "count": len(models)}

@app.get("/models/active", response_model=ActiveModelResponse)
async def get_active_model():
    """Return information about the currently loaded model."""
    if not _state.is_loaded:
        return ActiveModelResponse(loaded=False)
    return ActiveModelResponse(loaded=True, model=ModelInfo(**_state.model_info), device=_state.device)

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get a single model's details by UUID."""
    return _row_to_info(_query_model(model_id)).model_dump()

@app.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """Load a model into memory for serving."""
    row = _query_model(model_id)
    model_name = row["name"]
    model_dir = os.path.join(MODELS_DIR, model_name)
    model_path = os.path.join(model_dir, "model.pt")
    config_path = os.path.join(model_dir, "config.json")

    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found at {model_path}")

    config: Dict[str, Any] = {}
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    hidden_size = config.get("hidden_size", 768)
    intermediate_size = config.get("intermediate_size", 256)
    num_traits = config.get("num_traits", 5)
    dropout = config.get("dropout", 0.1)

    _state.clear()
    try:
        head = PersonalityPredictionHead(hidden_size, intermediate_size, num_traits, dropout)
        state_dict = torch.load(model_path, map_location=_state.device, weights_only=True)
        head.load_state_dict(state_dict)
        head.to(_state.device)
        head.eval()
    except Exception as exc:
        logger.error("Failed to load model weights: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")

    _state.model = head
    _state.config = config
    _state.model_info = _row_to_info(row).model_dump()
    logger.info("Model '%s' (id=%s) loaded on %s", model_name, model_id, _state.device)
    return {"status": "loaded", "model_id": model_id, "model_name": model_name, "device": _state.device}

@app.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload the currently loaded model if it matches model_id."""
    if not _state.is_loaded:
        raise HTTPException(status_code=400, detail="No model is currently loaded")
    if _state.model_info.get("id") != model_id:
        raise HTTPException(status_code=400, detail=f"Model {model_id} is not the currently loaded model")
    name = _state.model_info["name"]
    _state.clear()
    logger.info("Model '%s' (id=%s) unloaded", name, model_id)
    return {"status": "unloaded", "model_id": model_id, "model_name": name}

# -- Inference ---------------------------------------------------------------

def _require_model() -> None:
    if not _state.is_loaded:
        raise HTTPException(status_code=503,
                            detail="No model is loaded. Use POST /models/{id}/load first.")

def _run_inference(text: str) -> OceanScores:
    feat_size = _state.config.get("hidden_size", 768) if _state.config else 768
    features = extract_features(text, feature_size=feat_size).unsqueeze(0).to(_state.device)
    with torch.no_grad():
        output = _state.model(features)
    s = output.squeeze(0).cpu().tolist()
    return OceanScores(openness=round(s[0], 4), conscientiousness=round(s[1], 4),
                       extraversion=round(s[2], 4), agreeableness=round(s[3], 4),
                       neuroticism=round(s[4], 4))

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Return OCEAN personality scores for a single text input."""
    _require_model()
    start = time.perf_counter()
    scores = _run_inference(request.text)
    return PredictResponse(scores=scores, model_id=_state.model_info["id"],
                           model_name=_state.model_info["name"],
                           inference_time_ms=round((time.perf_counter() - start) * 1000, 2))

@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Return OCEAN personality scores for a batch of texts."""
    _require_model()
    start = time.perf_counter()
    preds = [_run_inference(t) for t in request.texts]
    return BatchPredictResponse(predictions=preds, model_id=_state.model_info["id"],
                                model_name=_state.model_info["name"], count=len(preds),
                                total_inference_time_ms=round((time.perf_counter() - start) * 1000, 2))

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: PredictRequest):
    """Full analysis: personality prediction, trait descriptions, and confidence."""
    _require_model()
    start = time.perf_counter()
    scores = _run_inference(request.text)
    score_map = scores.model_dump()
    traits: List[TraitDetail] = []
    for name in TRAIT_NAMES:
        val = score_map[name]
        label = "high" if val >= 0.5 else "low"
        traits.append(TraitDetail(
            trait=name, score=val, label=label,
            description=TRAIT_DESCRIPTIONS[name][label],
            confidence=round(abs(val - 0.5) * 2, 4),
        ))
    dominant = max(traits, key=lambda t: abs(t.score - 0.5)).trait
    return AnalyzeResponse(scores=scores, traits=traits, dominant_trait=dominant,
                           model_id=_state.model_info["id"], model_name=_state.model_info["name"],
                           inference_time_ms=round((time.perf_counter() - start) * 1000, 2))
