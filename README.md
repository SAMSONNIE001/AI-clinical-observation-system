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

The first version will use short self-simulated video clips recorded in controlled scenes. Each clip should contain one main incident or behaviour and one video-level label. These videos will represent safe observation scenarios such as sleeping, sitting, standing, walking, eating, reading, pacing, falling, aggressive movement simulation, attack-on-person placeholders, head-banging placeholders, property-damage placeholders, choking simulation, vomiting placeholders, ligature risk placeholders, visible bleeding simulations, sharp-object detection, and prolonged inactivity.

Ligature risk, attack on person, head banging, property damage, vomiting, visible bleeding, and sharp-object detection are included as high-risk system labels so alarm routing can be designed. They must not be recorded as real harm, real impact, real damage, real vomiting, or real violence. Self-harm instructional content remains outside the Version 1 dataset scope.

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

The detection endpoint defines the API contract that the later trained model
will plug into. When `video_path` is provided and
`ml/models/baseline_video_classifier.json` exists, the endpoint uses the
baseline classifier and returns `alarm_required` for high-risk predictions.

The first baseline training pipeline reads `dataset/training_manifest.jsonl` and
writes a simple OpenCV nearest-centroid classifier:

```text
python scripts/generate_dataset_metadata.py
python -m ml.training.train_baseline_classifier
```

This baseline is useful for exercising the full data flow. It is not the final
live-camera model.

The baseline can also be exercised against a live camera feed:

```text
python scripts/live_camera_detection.py
```

The live camera script records short rolling clips from the webcam, classifies
each clip with the combined clinical pipeline, displays the structured risk
group, risk level, behaviour prediction, and observation summary, and sounds a
simple alarm only when high-risk predictions meet the repeated confirmation
threshold. This is for testing the full workflow, not for clinical use.

The pretrained model path combines YOLO object detection, MediaPipe pose
movement analysis, and a later pretrained video/action classifier. YOLO
currently maps COCO `knife` and `scissors` detections into
`dangerous_objects_detected`, which can trigger the alarm contract:

```text
python scripts/test_pretrained_object_detector.py dataset/raw/sharp_object_detected/sharp_object_detected_001.mp4
```

MediaPipe pose movement can be tested separately:

```text
pip install -r backend/requirements-vision.txt
python scripts/test_mediapipe_pose.py dataset/raw/pacing/pacing_001.mp4
```

The combined pipeline can be tested with:

```text
python scripts/test_clinical_pipeline.py dataset/raw/pacing/pacing_001.mp4
```

The structured risk engine can be tested without loading YOLO or MediaPipe:

```text
python scripts/test_risk_engine.py
```

The first pretrained video/action classifier fine-tuning path uses TorchVision
`r3d_18` with a new project-specific classification head. Train grouped labels
first because the 18 detailed labels are too sparse for the current dataset:

```text
python -m ml.training.train_video_action_classifier --label-mode grouped --epochs 8 --batch-size 1 --unfreeze-backbone --learning-rate 0.00005 --patience 2
```

When `ml/models/video_action_grouped_classifier.pt` exists, the combined
pipeline uses it for risk-group classification. Until then, it falls back to the
baseline classifier.

Inspect the saved grouped checkpoint with:

```text
python scripts/inspect_video_action_checkpoint.py
```

YOLO weights may download the first time this command is run. The simulated
videos remain useful for fine-tuning and validation; they are not intended to
train the whole vision model from scratch. For final classification, a
pretrained video/action model is preferred over a plain CNN because many labels
depend on movement over time. See `docs/model_strategy.md` for the model
selection plan.

High-risk behaviours such as falls, choking simulation, vomiting, ligature risk,
visible bleeding, attack on person, and
sharp-object detection, head banging, and property damage set the detection
alarm contract so staff-facing workflows can trigger urgent review.

The observation note endpoint generates draft staff-review notes from a labelled
behaviour event. Notes are not final clinical records until reviewed by staff.

## Research Direction

Human-in-the-loop AI-assisted clinical observation for mental health and care settings using computer vision and automated documentation.
