from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.setup_checks import gather_startup_warnings


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="Backend API for AI-assisted clinical observation workflows.",
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.on_event("startup")
def on_startup() -> None:
    warnings = gather_startup_warnings()
    for warning in warnings:
        print(f"WARNING: {warning}")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "AI Clinical Observation System API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
