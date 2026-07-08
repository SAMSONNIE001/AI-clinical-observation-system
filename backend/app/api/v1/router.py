from fastapi import APIRouter

from app.api.v1.routes import dataset, detection, health, observation_notes, videos


api_router = APIRouter()
api_router.include_router(dataset.router, tags=["dataset"])
api_router.include_router(detection.router, tags=["detection"])
api_router.include_router(health.router, tags=["health"])
api_router.include_router(observation_notes.router, tags=["observation-notes"])
api_router.include_router(videos.router, tags=["videos"])
