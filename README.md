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

The first version will use short self-simulated video clips recorded in controlled scenes. Each clip should contain one main incident or behaviour and one video-level label. These videos will represent safe observation scenarios such as sleeping, sitting, standing, walking, eating, reading, pacing, falling, aggressive movement simulation, fighting or attack placeholders, head-banging placeholders, property-damage placeholders, choking simulation, vomiting placeholders, ligature risk placeholders, cutting-risk placeholders, visible blood or bleeding simulations, sharp-object detection, and prolonged inactivity.

Ligature risk, fighting, attack on person, head banging, property damage, vomiting, cutting risk, visible blood, visible bleeding, and sharp-object detection are included as high-risk system labels so alarm routing can be designed. They must not be recorded as real harm, real impact, real damage, real vomiting, or real violence. Self-harm instructional content remains outside the Version 1 dataset scope.

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
- `POST /api/v1/dataset/exports/training-manifest`
- `POST /api/v1/detection/predict`
- `POST /api/v1/observation-notes/generate`

The video upload endpoint accepts `.mp4`, `.avi`, `.mov`, and `.mkv` files and stores them locally under backend storage for later dataset management and behaviour detection.

Dataset endpoints register video-level labels for self-simulated videos. Time
range annotations are available for later Version 2 work, but they are optional
for the first training dataset.

The training manifest export creates a JSONL file from the video-level labels so
future model training can consume the dataset consistently.

Before recording simulated clips, use `docs/simulated_video_recording_plan.md`
and `dataset/metadata_template.csv` to keep filenames, labels, and metadata
consistent across the first pilot batch.

Raw pilot videos are organised by behaviour under `dataset/raw`, for example
`dataset/raw/pacing/pacing_001.mp4`.

The detection endpoint currently returns a stub response. It defines the API
contract that the later trained model will plug into.

The first baseline training pipeline reads `dataset/training_manifest.jsonl` and
writes a simple OpenCV nearest-centroid classifier:

```text
python scripts/generate_dataset_metadata.py
python -m ml.training.train_baseline_classifier
```

This baseline is useful for exercising the full data flow. It is not the final
live-camera model.

Deferred high-risk labels such as ligature risk, cutting risk, fighting, attack
on person, head banging, property damage, blood/bleeding, and sharp-object
detection should not be added to training until safe simulations or approved
footage are available.

High-risk behaviours such as falls, choking simulation, vomiting, ligature risk, cutting
risk, visible blood, visible bleeding, fighting, attack on person, and
sharp-object detection, head banging, and property damage set the detection
alarm contract so staff-facing workflows can trigger urgent review.

The observation note endpoint generates draft staff-review notes from a labelled
behaviour event. Notes are not final clinical records until reviewed by staff.

## Research Direction

Human-in-the-loop AI-assisted clinical observation for mental health and care settings using computer vision and automated documentation.
