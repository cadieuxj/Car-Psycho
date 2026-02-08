"""
Data Service - Dataset Management API
Production-ready service for dataset registration, scanning, statistics,
and sample preview for psychometric car-personality datasets.
"""

import csv
import json
import os
import shutil
import statistics
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv("DATABASE_URL", "")
DATA_ROOT: Path = Path("/app/data")
SUPPORTED_FORMATS: set[str] = {".jsonl", ".csv", ".json"}
OCEAN_TRAITS: list[str] = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TraitStats(BaseModel):
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0


class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int


class TraitDistribution(BaseModel):
    stats: TraitStats
    histogram: list[HistogramBin]


class DatasetSummary(BaseModel):
    id: str
    name: str
    dataset_type: Optional[str] = None
    source: Optional[str] = None
    description: Optional[str] = None
    file_path: Optional[str] = None
    format: Optional[str] = None
    num_samples: int = 0
    is_processed: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    total_samples_in_db: int = 0
    trait_means: dict[str, float] = Field(default_factory=dict)


class DatasetDetail(DatasetSummary):
    schema_info: Optional[dict[str, Any]] = None
    stats: Optional[dict[str, Any]] = None


class SampleOut(BaseModel):
    id: str
    sample_data: Optional[dict[str, Any]] = None
    personality_labels: Optional[dict[str, Any]] = None
    car_profile: Optional[dict[str, Any]] = None
    source_text: Optional[str] = None
    is_synthetic: bool = False
    created_at: Optional[datetime] = None


class DatasetStatsResponse(BaseModel):
    dataset_id: str
    total_samples: int
    trait_distributions: dict[str, TraitDistribution]
    source_breakdown: dict[str, int]


class ScanResult(BaseModel):
    scanned: int
    registered: int
    skipped: int
    details: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    service: str = "data"
    database: str
    data_directory: str
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float


class UploadResponse(BaseModel):
    id: str
    name: str
    num_samples: int
    message: str


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Car-Psycho Data Service",
    description="Dataset management API for psychometric car-personality data",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

psycopg2.extras.register_uuid()


@contextmanager
def get_conn():
    """Yield a psycopg2 connection, commit on success, rollback on error."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _row_to_dict(cur) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# File inspection helpers
# ---------------------------------------------------------------------------


def _inspect_jsonl(path: Path) -> tuple[int, dict | None]:
    """Return (line_count, first_record_schema) for a JSONL file."""
    count = 0
    schema: dict | None = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            count += 1
            if schema is None:
                try:
                    record = json.loads(line)
                    schema = {k: type(v).__name__ for k, v in record.items()}
                except json.JSONDecodeError:
                    pass
    return count, schema


def _inspect_csv(path: Path) -> tuple[int, dict | None]:
    count = 0
    schema: dict | None = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        headers = next(reader, None)
        if headers:
            schema = {h: "str" for h in headers}
        for _ in reader:
            count += 1
    return count, schema


def _inspect_json(path: Path) -> tuple[int, dict | None]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        schema = (
            {k: type(v).__name__ for k, v in data[0].items()}
            if data and isinstance(data[0], dict)
            else None
        )
        return len(data), schema
    return 1, {k: type(v).__name__ for k, v in data.items()} if isinstance(data, dict) else None


def _extract_ocean_values(path: Path, fmt: str) -> dict[str, list[float]]:
    """Extract OCEAN trait values from a dataset file for stat computation."""
    values: dict[str, list[float]] = {t: [] for t in OCEAN_TRAITS}
    records: list[dict] = []

    try:
        if fmt == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        elif fmt == ".json":
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
                records = data if isinstance(data, list) else [data]
        elif fmt == ".csv":
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                records = list(reader)
    except Exception:
        return values

    for rec in records:
        labels = rec.get("personality_labels") or rec
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except (json.JSONDecodeError, TypeError):
                continue
        for trait in OCEAN_TRAITS:
            val = labels.get(trait)
            if val is not None:
                try:
                    values[trait].append(float(val))
                except (ValueError, TypeError):
                    continue
    return values


def _compute_trait_stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    return {
        "mean": round(statistics.mean(vals), 4),
        "std": round(statistics.pstdev(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "median": round(statistics.median(vals), 4),
    }


def _compute_histogram(vals: list[float], bins: int = 10) -> list[dict]:
    hist: list[dict] = []
    for i in range(bins):
        lo = round(i / bins, 2)
        hi = round((i + 1) / bins, 2)
        count = sum(1 for v in vals if lo <= v < hi) if i < bins - 1 else sum(1 for v in vals if lo <= v <= hi)
        hist.append({"bin_start": lo, "bin_end": hi, "count": count})
    return hist


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "data", "status": "running", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with DB connection, data directory, and disk info."""
    db_status = "ok"
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:
        db_status = f"error: {exc}"

    data_dir_status = "ok" if DATA_ROOT.is_dir() else "missing"

    try:
        usage = shutil.disk_usage(DATA_ROOT if DATA_ROOT.exists() else "/")
    except OSError:
        usage = shutil.disk_usage("/")

    return HealthResponse(
        status="healthy" if db_status == "ok" else "degraded",
        database=db_status,
        data_directory=data_dir_status,
        disk_total_gb=round(usage.total / (1024**3), 2),
        disk_used_gb=round(usage.used / (1024**3), 2),
        disk_free_gb=round(usage.free / (1024**3), 2),
    )


# ---- Dataset CRUD ----------------------------------------------------------


@app.get("/datasets", response_model=list[DatasetSummary])
async def list_datasets():
    """List all datasets with summary stats."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT d.*,
                           COALESCE(s.cnt, 0) AS total_samples_in_db
                    FROM datasets d
                    LEFT JOIN (
                        SELECT dataset_id, COUNT(*) AS cnt
                        FROM data_samples GROUP BY dataset_id
                    ) s ON s.dataset_id = d.id
                    ORDER BY d.created_at DESC
                    """
                )
                rows = _row_to_dict(cur)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    results: list[DatasetSummary] = []
    for r in rows:
        trait_means: dict[str, float] = {}
        stats_blob = r.get("stats")
        if isinstance(stats_blob, dict):
            for trait in OCEAN_TRAITS:
                t_info = stats_blob.get(trait)
                if isinstance(t_info, dict) and "mean" in t_info:
                    trait_means[trait] = t_info["mean"]

        results.append(
            DatasetSummary(
                id=str(r["id"]),
                name=r["name"],
                dataset_type=r.get("dataset_type"),
                source=r.get("source"),
                description=r.get("description"),
                file_path=r.get("file_path"),
                format=r.get("format"),
                num_samples=r.get("num_samples") or 0,
                is_processed=r.get("is_processed", False),
                created_at=r.get("created_at"),
                updated_at=r.get("updated_at"),
                total_samples_in_db=r.get("total_samples_in_db", 0),
                trait_means=trait_means,
            )
        )
    return results


@app.get("/datasets/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: str):
    """Get full dataset details including schema and stats."""
    try:
        uid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset ID format")

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT d.*,
                           COALESCE(s.cnt, 0) AS total_samples_in_db
                    FROM datasets d
                    LEFT JOIN (
                        SELECT dataset_id, COUNT(*) AS cnt
                        FROM data_samples GROUP BY dataset_id
                    ) s ON s.dataset_id = d.id
                    WHERE d.id = %s
                    """,
                    (uid,),
                )
                rows = _row_to_dict(cur)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    if not rows:
        raise HTTPException(status_code=404, detail="Dataset not found")

    r = rows[0]
    trait_means: dict[str, float] = {}
    stats_blob = r.get("stats")
    if isinstance(stats_blob, dict):
        for trait in OCEAN_TRAITS:
            t_info = stats_blob.get(trait)
            if isinstance(t_info, dict) and "mean" in t_info:
                trait_means[trait] = t_info["mean"]

    return DatasetDetail(
        id=str(r["id"]),
        name=r["name"],
        dataset_type=r.get("dataset_type"),
        source=r.get("source"),
        description=r.get("description"),
        file_path=r.get("file_path"),
        format=r.get("format"),
        num_samples=r.get("num_samples") or 0,
        is_processed=r.get("is_processed", False),
        created_at=r.get("created_at"),
        updated_at=r.get("updated_at"),
        total_samples_in_db=r.get("total_samples_in_db", 0),
        trait_means=trait_means,
        schema_info=r.get("schema_info"),
        stats=r.get("stats"),
    )


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Remove dataset registration from DB (does NOT delete the file)."""
    try:
        uid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset ID format")

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM datasets WHERE id = %s RETURNING id", (uid,))
                deleted = cur.fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {"detail": "Dataset registration removed", "id": dataset_id}


# ---- Preview & Stats -------------------------------------------------------


@app.get("/datasets/{dataset_id}/preview", response_model=list[SampleOut])
async def preview_dataset(dataset_id: str):
    """Return first 20 samples from a dataset."""
    try:
        uid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset ID format")

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM datasets WHERE id = %s", (uid,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Dataset not found")

                cur.execute(
                    """
                    SELECT id, sample_data, personality_labels, car_profile,
                           source_text, is_synthetic, created_at
                    FROM data_samples
                    WHERE dataset_id = %s
                    ORDER BY created_at ASC
                    LIMIT 20
                    """,
                    (uid,),
                )
                rows = _row_to_dict(cur)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    return [
        SampleOut(
            id=str(r["id"]),
            sample_data=r.get("sample_data"),
            personality_labels=r.get("personality_labels"),
            car_profile=r.get("car_profile"),
            source_text=r.get("source_text"),
            is_synthetic=r.get("is_synthetic", False),
            created_at=r.get("created_at"),
        )
        for r in rows
    ]


@app.get("/datasets/{dataset_id}/stats", response_model=DatasetStatsResponse)
async def dataset_stats(dataset_id: str):
    """Detailed statistics: OCEAN trait distributions, histograms, source breakdown."""
    try:
        uid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset ID format")

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM datasets WHERE id = %s", (uid,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Dataset not found")

                cur.execute(
                    "SELECT personality_labels FROM data_samples WHERE dataset_id = %s",
                    (uid,),
                )
                label_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT COALESCE(sample_data->>'source', 'unknown') AS src,
                           COUNT(*) AS cnt
                    FROM data_samples
                    WHERE dataset_id = %s
                    GROUP BY src
                    """,
                    (uid,),
                )
                source_rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    # Collect trait values
    trait_values: dict[str, list[float]] = {t: [] for t in OCEAN_TRAITS}
    for (labels,) in label_rows:
        if not isinstance(labels, dict):
            continue
        for trait in OCEAN_TRAITS:
            val = labels.get(trait)
            if val is not None:
                try:
                    trait_values[trait].append(float(val))
                except (ValueError, TypeError):
                    continue

    distributions: dict[str, TraitDistribution] = {}
    for trait in OCEAN_TRAITS:
        vals = trait_values[trait]
        distributions[trait] = TraitDistribution(
            stats=TraitStats(**_compute_trait_stats(vals)),
            histogram=[HistogramBin(**b) for b in _compute_histogram(vals)],
        )

    source_breakdown = {row[0]: row[1] for row in source_rows}

    return DatasetStatsResponse(
        dataset_id=dataset_id,
        total_samples=len(label_rows),
        trait_distributions=distributions,
        source_breakdown=source_breakdown,
    )


# ---- Scan & Upload ---------------------------------------------------------


@app.post("/datasets/scan", response_model=ScanResult)
async def scan_datasets():
    """Scan /app/data/ recursively for new dataset files and register them."""
    if not DATA_ROOT.is_dir():
        raise HTTPException(status_code=500, detail=f"Data directory {DATA_ROOT} not found")

    # Gather known file paths from DB
    known_paths: set[str] = set()
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT file_path FROM datasets WHERE file_path IS NOT NULL")
                known_paths = {r[0] for r in cur.fetchall()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    scanned = 0
    registered = 0
    skipped = 0
    details: list[dict[str, Any]] = []

    for root, _dirs, files in os.walk(DATA_ROOT):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in SUPPORTED_FORMATS:
                continue

            full_path = os.path.join(root, fname)
            scanned += 1

            if full_path in known_paths:
                skipped += 1
                details.append({"file": full_path, "action": "skipped", "reason": "already registered"})
                continue

            # Inspect the file
            try:
                if ext == ".jsonl":
                    num_samples, schema = _inspect_jsonl(Path(full_path))
                elif ext == ".csv":
                    num_samples, schema = _inspect_csv(Path(full_path))
                elif ext == ".json":
                    num_samples, schema = _inspect_json(Path(full_path))
                else:
                    continue
            except Exception as exc:
                skipped += 1
                details.append({"file": full_path, "action": "skipped", "reason": str(exc)})
                continue

            # Compute OCEAN stats from file
            ocean_vals = _extract_ocean_values(Path(full_path), ext)
            file_stats: dict[str, Any] = {}
            for trait in OCEAN_TRAITS:
                if ocean_vals[trait]:
                    file_stats[trait] = _compute_trait_stats(ocean_vals[trait])

            # Determine dataset_type from subdirectory
            rel = os.path.relpath(full_path, DATA_ROOT)
            parts = Path(rel).parts
            dataset_type = parts[0] if len(parts) > 1 else "unknown"

            new_id = uuid.uuid4()
            now = datetime.now(timezone.utc)

            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO datasets
                                (id, name, dataset_type, source, file_path, format,
                                 num_samples, schema_info, stats, is_processed,
                                 created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, %s, %s)
                            """,
                            (
                                new_id,
                                fname,
                                dataset_type,
                                dataset_type,
                                full_path,
                                ext.lstrip("."),
                                num_samples,
                                json.dumps(schema),
                                json.dumps(file_stats),
                                now,
                                now,
                            ),
                        )
            except Exception as exc:
                skipped += 1
                details.append({"file": full_path, "action": "error", "reason": str(exc)})
                continue

            registered += 1
            details.append({
                "file": full_path,
                "action": "registered",
                "id": str(new_id),
                "num_samples": num_samples,
                "format": ext.lstrip("."),
            })

    return ScanResult(scanned=scanned, registered=registered, skipped=skipped, details=details)


@app.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a JSONL file as a new dataset."""
    if not file.filename or not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported")

    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    # Validate and count lines
    lines = [ln for ln in text.splitlines() if ln.strip()]
    num_samples = 0
    schema: dict | None = None
    ocean_vals: dict[str, list[float]] = {t: [] for t in OCEAN_TRAITS}

    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        num_samples += 1
        if schema is None and isinstance(record, dict):
            schema = {k: type(v).__name__ for k, v in record.items()}
        labels = record.get("personality_labels", record)
        if isinstance(labels, dict):
            for trait in OCEAN_TRAITS:
                val = labels.get(trait)
                if val is not None:
                    try:
                        ocean_vals[trait].append(float(val))
                    except (ValueError, TypeError):
                        continue

    if num_samples == 0:
        raise HTTPException(status_code=400, detail="File contains no valid JSONL records")

    # Persist file
    dest_dir = DATA_ROOT / "raw"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename
    suffix = 1
    while dest_path.exists():
        stem = Path(file.filename).stem
        dest_path = dest_dir / f"{stem}_{suffix}.jsonl"
        suffix += 1

    with open(dest_path, "wb") as fh:
        fh.write(content)

    file_stats = {}
    for trait in OCEAN_TRAITS:
        if ocean_vals[trait]:
            file_stats[trait] = _compute_trait_stats(ocean_vals[trait])

    new_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO datasets
                        (id, name, dataset_type, source, file_path, format,
                         num_samples, schema_info, stats, is_processed,
                         created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, %s, %s)
                    """,
                    (
                        new_id,
                        dest_path.name,
                        "raw",
                        "upload",
                        str(dest_path),
                        "jsonl",
                        num_samples,
                        json.dumps(schema),
                        json.dumps(file_stats),
                        now,
                        now,
                    ),
                )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    return UploadResponse(
        id=str(new_id),
        name=dest_path.name,
        num_samples=num_samples,
        message="Dataset uploaded and registered successfully",
    )
