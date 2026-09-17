"""Local FastAPI service exposing the fitted hero-recommendation pipeline.

The pipeline artifact (pipeline.joblib) is loaded exactly once at module
import time. If it cannot be loaded for any reason (missing, corrupted,
incompatible sklearn version, etc.), the module still imports successfully
and the app still starts - only the endpoints that need the artifact report
HTTP 503 instead of crashing. This module never rebuilds the artifact;
that remains build_pipeline.py's job.
"""

import logging
from typing import List, Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import pipeline_def  # noqa: F401  # required so joblib can resolve HeroRecommenderTransformer

logger = logging.getLogger("uvicorn.error")

PIPELINE_PATH = "pipeline.joblib"
SERVICE_NAME = "overwatch-hero-recommender"

RankLiteral = Literal[
    "Bronze", "Silver", "Gold", "Platinum", "Emerald", "Diamond", "Master", "Grandmaster"
]
InputLiteral = Literal["PC", "Console"]
RegionLiteral = Literal["Americas", "Europe", "Asia"]

app = FastAPI(title=SERVICE_NAME)

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


class RecommendRequest(BaseModel):
    heroes: List[str] = Field(..., min_length=1, max_length=2)
    rank: RankLiteral
    input: InputLiteral
    region: RegionLiteral

    @field_validator("heroes")
    @classmethod
    def validate_heroes(cls, heroes: List[str]) -> List[str]:
        if len(set(heroes)) != len(heroes):
            raise ValueError("duplicate hero IDs are not allowed")

        # Hero-ID-vs-Damage-roster validation depends on the loaded artifact.
        # If the artifact isn't loaded, skip this check here and let the
        # endpoint itself report 503 - a missing artifact is a service
        # problem, not a client validation problem.
        if _bundle is not None:
            damage_heroes = set(_bundle["damage_heroes"])
            unknown = [h for h in heroes if h not in damage_heroes]
            if unknown:
                raise ValueError(f"unknown Damage hero id(s): {unknown}")

        return heroes


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


@app.post("/recommend")
def recommend(request: RecommendRequest):
    if _bundle is None:
        raise HTTPException(status_code=503, detail=_unavailable_detail())

    pipeline = _bundle["pipeline"]
    result = pipeline.transform([request.model_dump()])
    return result[0]
