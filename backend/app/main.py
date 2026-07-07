from fastapi import FastAPI


app = FastAPI(title="AI Clinical Observation System")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "AI Clinical Observation System API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
