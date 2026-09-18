"""Local FastAPI service exposing the fitted hero-recommendation pipeline.

The pipeline artifact (pipeline.joblib) is loaded exactly once at module
import time. If it cannot be loaded for any reason (missing, corrupted,
incompatible sklearn version, etc.), the module still imports successfully
and the app still starts - only the endpoints that need the artifact report
HTTP 503 instead of crashing. This module never rebuilds the artifact;
that remains build_pipeline.py's job.
"""

import logging
from typing import Literal, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator

import pipeline_def  # noqa: F401  # required so joblib can resolve HeroRecommenderTransformer

logger = logging.getLogger("uvicorn.error")

PIPELINE_PATH = "pipeline.joblib"
SERVICE_NAME = "overwatch-hero-recommender"

RankLiteral = Literal[
    "Bronze", "Silver", "Gold", "Platinum", "Emerald", "Diamond", "Master", "Grandmaster"
]
InputLiteral = Literal["PC", "Console"]
RegionLiteral = Literal["Americas", "Europe", "Asia"]
RoleLiteral = Literal["Tank", "Damage", "Support"]

app = FastAPI(title=SERVICE_NAME)

# The frontend (a static site, eventually hosted on Vercel) calls this API
# directly from the browser. This is a public, read-only recommendation
# endpoint with no auth/cookies/session state, so allowing any origin is
# sufficient for this assignment rather than hardcoding a specific Vercel
# domain that doesn't exist yet. This does not weaken request validation -
# Pydantic still rejects invalid bodies with 422 regardless of origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_bundle = None
_load_error = None


def _load_bundle():
    global _bundle, _load_error
    try:
        _bundle = joblib.load(PIPELINE_PATH)
        _load_error = None
        logger.info("Loaded pipeline artifact from %s", PIPELINE_PATH)
    except Exception as exc:  # noqa: BLE001 - any load failure must not crash the module
        _bundle = None
        _load_error = f"{type(exc).__name__}: {exc}"
        logger.error("Failed to load pipeline artifact from %s: %s", PIPELINE_PATH, _load_error)


_load_bundle()


def _unavailable_detail():
    return {
        "service": SERVICE_NAME,
        "status": "unavailable",
        "artifact_loaded": False,
        "message": f"pipeline artifact could not be loaded from '{PIPELINE_PATH}': {_load_error}",
    }


class PoolRequest(BaseModel):
    """Complete-pool recommendation request.

    `secondary` omitted/null -> Pool Builder (recommend two backups around
    `main`). `secondary` supplied -> Pool Completer (recommend one tertiary
    that best completes {main, secondary}). One endpoint, one schema, mode
    selected by whether `secondary` is present - this intentionally replaces
    the old `heroes: List[str]` sequential-slot schema, which encoded a
    misleading Hero#2/Hero#3 ordering that the complete-pool architecture no
    longer has (Builder's two backups are an unordered pair). The old
    schema/behavior remains fully available in git history and via
    HeroRecommenderTransformer._recommend_for_request for direct comparison.
    """

    role: RoleLiteral
    main: str
    secondary: Optional[str] = None
    rank: RankLiteral
    input: InputLiteral
    region: RegionLiteral

    @model_validator(mode="after")
    def validate_main_secondary(self):
        if self.secondary is not None and self.secondary == self.main:
            raise ValueError("main and secondary must be distinct heroes")

        # Hero-ID-vs-role validation depends on the loaded artifact. If the
        # artifact isn't loaded, skip this check here and let the endpoint
        # itself report 503 - a missing artifact is a service problem, not a
        # client validation problem.
        if _bundle is not None:
            eligible = set(_bundle["heroes_by_role"].get(self.role.upper(), []))
            if self.main not in eligible:
                raise ValueError(f"hero id {self.main!r} is not valid for role {self.role!r}")
            if self.secondary is not None and self.secondary not in eligible:
                raise ValueError(f"hero id {self.secondary!r} is not valid for role {self.role!r}")
        return self


@app.get("/")
def root():
    if _bundle is None:
        raise HTTPException(status_code=503, detail=_unavailable_detail())

    metadata = _bundle["metadata"]
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "artifact_loaded": True,
        "pipeline_steps": metadata["steps"],
        "built_at": metadata["built_at"],
        "sklearn_version": metadata["sklearn_version"],
    }


@app.get("/health")
def health():
    """Lightweight liveness/health check, distinct from /pipeline's fuller
    artifact metadata. Uses the already-loaded bundle - no re-loading, no
    recommendation logic."""
    if _bundle is None:
        raise HTTPException(status_code=503, detail=_unavailable_detail())

    return {
        "service": SERVICE_NAME,
        "status": "healthy",
        "artifact_loaded": True,
    }


@app.get("/pipeline")
def pipeline_info():
    """Artifact/pipeline metadata, exposed separately from /health so the two
    concerns (is the service up vs. what pipeline is it running) are
    independently testable."""
    if _bundle is None:
        raise HTTPException(status_code=503, detail=_unavailable_detail())

    metadata = _bundle["metadata"]
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "artifact_loaded": True,
        "pipeline_steps": metadata["steps"],
        "built_at": metadata["built_at"],
        "sklearn_version": metadata["sklearn_version"],
        "python_version": metadata.get("python_version"),
    }


@app.post("/recommend")
def recommend(request: PoolRequest):
    """Complete-pool recommendation. `secondary` omitted -> Builder mode
    (recommend two backups). `secondary` supplied -> Completer mode
    (recommend one tertiary). See PoolRequest for the mode-selection rule."""
    if _bundle is None:
        raise HTTPException(status_code=503, detail=_unavailable_detail())

    pipeline = _bundle["pipeline"]
    payload = request.model_dump()
    # heroes_by_role_ keys come from rank_df's role column (Blizzard's own
    # "TANK"/"DAMAGE"/"SUPPORT" casing); the API accepts the friendlier
    # Title-case values, normalized here at the boundary.
    payload["role"] = payload["role"].upper()
    result = pipeline.transform([payload])
    return result[0]
