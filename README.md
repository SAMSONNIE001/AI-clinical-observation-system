# AI-Assisted Clinical Observation System

This project explores the use of computer vision to support clinical observation in mental health units, care homes, and supported living services.

The system is designed to assist staff by detecting selected behaviours from video, generating alerts, and producing draft observation notes for staff review.

## Aim

The goal is not to replace healthcare workers, but to support safer and more efficient observation practice through human-in-the-loop AI.

## Planned Features

- Behaviour detection from simulated video scenes
- Fall, pacing, inactivity, sleeping, and distress-pattern detection
- Staff alert dashboard
- AI-generated draft observation notes
- Human review and approval workflow
- Simulated behavioural dataset for research

## Dataset

The first version will use short self-simulated video clips recorded in controlled scenes. Each clip should contain one main incident or behaviour and one video-level label. These videos will represent safe observation scenarios such as sleeping, sitting, standing, walking, eating, reading, pacing, falling, aggressive movement simulation, choking simulation, and prolonged inactivity.

Highly sensitive behaviours such as ligature attempts and self-harm are outside the Version 1 dataset scope. They should only be revisited after ethics, safeguarding, simulation design, and clinical review requirements are clear.

## Tech Stack

- Python
- FastAPI
- OpenCV
- MediaPipe or YOLO
- PostgreSQL
- React
- GitHub Actions

## Project Structure

```text
AI-Clinical-Observation-System
├── backend
│   ├── app
│   │   ├── api
│   │   ├── core
│   │   ├── domain
│   │   ├── models
│   │   ├── schemas
│   │   ├── services
│   │   ├── utils
│   │   └── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend
├── ml
│   ├── training
│   ├── inference
│   ├── datasets
│   └── models
├── dataset
└── docs
```

## Build Order

1. Project architecture
2. Video upload API
3. Behaviour detection pipeline
4. Dataset management
5. Observation note generation
6. Real-time monitoring
7. Staff dashboard
8. Model training with simulated dataset

The first phase builds the system that the AI components will plug into. Model training comes later, after the API and data flow are stable.

## Current API

- `GET /health`
- `GET /api/v1/health`
- `POST /api/v1/videos/upload`
- `POST /api/v1/dataset/videos`
- `GET /api/v1/dataset/videos`
- `POST /api/v1/dataset/annotations`
- `GET /api/v1/dataset/annotations/{video_id}`
- `POST /api/v1/detection/predict`
- `POST /api/v1/observation-notes/generate`

The video upload endpoint accepts `.mp4`, `.avi`, `.mov`, and `.mkv` files and stores them locally under backend storage for later dataset management and behaviour detection.

Dataset endpoints register video-level labels for self-simulated videos. Time
range annotations are available for later Version 2 work, but they are optional
for the first training dataset.

The detection endpoint currently returns a stub response. It defines the API
contract that the later trained model will plug into.

The observation note endpoint generates draft staff-review notes from a labelled
behaviour event. Notes are not final clinical records until reviewed by staff.

## Research Direction

Human-in-the-loop AI-assisted clinical observation for mental health and care settings using computer vision and automated documentation.
