# Simulated Dataset Design

Version 1 of the dataset focuses on behaviours that can be recorded safely in a
controlled volunteer setting.

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

## Annotation Fields

Each labelled behaviour segment should include:

- `video_id`
- `behaviour`
- `start_time_seconds`
- `end_time_seconds`
- `label_quality`
- `annotated_by`
- `notes`

The annotation time range identifies where the behaviour occurs inside a video.
This is the information the later training pipeline will need for model tuning.
