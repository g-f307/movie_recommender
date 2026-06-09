from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from cinebot_ml.config import DEFAULT_DATA_PATH, DEFAULT_TOP_N
from cinebot_ml.dataset import (
    canonicalize_decade_preference,
    canonicalize_genre,
    canonicalize_popularity_preference,
)
from cinebot_ml.predictor import CinebotRecommender

app = FastAPI(title="CineBot ML API", version="1.0.0")
recommender = CinebotRecommender()


class PredictionRequest(BaseModel):
    ranked_genres: list[str] = Field(min_items=3, max_items=3)
    decade_preference: str
    popularity_preference: str
    data_path: str | None = str(DEFAULT_DATA_PATH)
    top_n: int = Field(default=DEFAULT_TOP_N, ge=1, le=10)
    user_id: str | int | None = None

    @field_validator("ranked_genres")
    @classmethod
    def validate_ranked_genres(cls, value: list[str]) -> list[str]:
        canonical = [canonicalize_genre(item) for item in value]
        if len(set(canonical)) != 3:
            raise ValueError("Os 3 gêneros precisam ser distintos.")
        return canonical

    @field_validator("decade_preference")
    @classmethod
    def validate_decade_preference(cls, value: str) -> str:
        canonical = canonicalize_decade_preference(value)
        if canonical is None:
            raise ValueError("Década inválida.")
        return canonical

    @field_validator("popularity_preference")
    @classmethod
    def validate_popularity_preference(cls, value: str) -> str:
        canonical = canonicalize_popularity_preference(value)
        if canonical is None:
            raise ValueError("Popularidade inválida.")
        return canonical


@app.get("/saude")
def saude() -> dict:
    return recommender.healthcheck()


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    try:
        return recommender.recommend(
            ranked_genres=payload.ranked_genres,
            decade_preference=payload.decade_preference,
            popularity_preference=payload.popularity_preference,
            data_path=Path(payload.data_path) if payload.data_path else DEFAULT_DATA_PATH,
            top_n=payload.top_n,
            user_id=payload.user_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
