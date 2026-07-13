# AI Clinical Observation Dashboard

This is a minimal static dashboard prototype for the backend API.

## Local usage

1. Start the backend API:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. Serve the frontend files:

```bash
cd frontend
python -m http.server 8080
```

3. Open the static dashboard:

```text
http://localhost:8080
```

## Notes

- The dashboard calls `http://localhost:8000/api/v1`.
- If the backend runs on a different host or port, update `frontend/app.js`.
