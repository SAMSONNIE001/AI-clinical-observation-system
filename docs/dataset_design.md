# Simulated Dataset Design

Version 1 of the dataset focuses on short self-simulated videos recorded safely
in a controlled volunteer setting.

The Version 1 labelling rule is:

```text
One short video = one primary behaviour label
```

Each clip should contain one main incident or behaviour from start to finish. For
example, `pacing_001.mp4` should be labelled `pacing`, and `fall_001.mp4` should
be labelled `fall`.

## Behaviour Classes

| Category | Behaviour |
| --- | --- |
| Normal | Sleeping |
| Normal | Sitting |
| Normal | Standing |
| Normal | Walking |
| Normal | Eating |
| Normal | Reading |
| Risk | Pacing |
| Risk | Fall |
| Risk | Aggressive movement |
| Risk | Choking simulation |
| Risk | Prolonged inactivity |

Highly sensitive behaviours such as ligature attempts and self-harm are outside
the Version 1 dataset scope. They should only be revisited after ethics,
safeguarding, simulation design, and clinical review requirements are clear.

## Metadata Fields

Each uploaded simulated video should be registered with:

- `video_id`
- `filename`
- `behaviour_type`
- `category`
- `scenario_name`
- `duration_seconds`
- `environment`
- `camera_angle`
- `recorded_by`
- `recorded_at`
- `notes`

## Version 1 Labelling

For Version 1 model training, the video metadata provides the training label:

- `video_id`
- `filename`
- `behaviour_type`
- `category`
- `scenario_name`
- `duration_seconds`
- `environment`

This avoids the overhead of manually timestamping each incident while the
dataset is still small and controlled.

## Optional Version 2 Annotation Fields

Time-range annotations are optional and should be used later if a single video
contains multiple behaviours or if we need frame-level evaluation.

Each labelled behaviour segment can include:

- `video_id`
- `behaviour`
- `start_time_seconds`
- `end_time_seconds`
- `label_quality`
- `annotated_by`
- `notes`

The annotation time range identifies where a behaviour occurs inside a longer or
mixed-behaviour video. It is not required for the Version 1 one-label-per-video
training workflow.

## Detection Pipeline Status

The backend includes a stub detection endpoint so the API contract exists before
model training starts. It does not infer behaviour yet. The real model will be
connected after the Version 1 labelled clips are collected and prepared for
training.
