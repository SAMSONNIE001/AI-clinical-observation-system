from fastapi import APIRouter

from app.api.v1.routes import dataset, health, videos


api_router = APIRouter()
api_router.include_router(dataset.router, tags=["dataset"])
api_router.include_router(health.router, tags=["health"])
api_router.include_router(videos.router, tags=["videos"])
