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
| Risk | Attack on person |
| Risk | Head banging |
| Risk | Property damage |
| Risk | Choking simulation |
| Risk | Vomiting |
| Risk | Ligature risk |
| Risk | Bleeding visible |
| Risk | Sharp object detected |
| Risk | Prolonged inactivity |

Ligature risk is included as a high-risk system label so the product contract
can route it to urgent alarms. It must not be recorded as a real attempt. Any
Version 1 example should be a safe, reviewed simulation or placeholder asset
only after ethics, safeguarding, simulation design, and clinical review
requirements are clear.

Visible bleeding and sharp-object detection are included as high-risk safety
labels. They must not involve real injury. Any dataset examples should use safe
props, fake blood, controlled acting, or licensed/approved footage, and should
be reviewed before use. Self-harm instructional content remains outside the
Version 1 dataset scope.

Attack-on-person is the single label for fighting, patient-on-patient
aggression, and attack-on-staff simulations. It is included for urgent safety
escalation. It should not involve real violence. Any examples should use
controlled acting with no contact, staged movement, or approved/licensed
footage.

Head banging and property damage labels are included for urgent safety
escalation. They should not involve real impact, real injury, or real damage.
Any examples should use controlled acting, soft props, staged movement, or
approved/licensed footage.

Vomiting is included as a high-risk safety label because it can indicate acute
illness, choking risk, or aspiration risk. It must not involve real vomiting or
unsafe ingestion. Any examples should use safe acting, props, or
approved/licensed footage.

## Alarm Rules

The system should sound an alarm for high-risk behaviours:

- `fall`
- `attack_on_person`
- `head_banging`
- `property_damage`
- `choking_simulation`
- `vomiting`
- `ligature_risk`
- `bleeding_visible`
- `sharp_object_detected`

The system should also sound an alarm if dangerous objects are detected,
including:

- `blade`
- `box_cutter`
- `broken_glass`
- `knife`
- `razor`
- `scissors`
- `sharp_object`

Medium-risk behaviours such as `pacing`, `aggressive_movement`, and
`prolonged_inactivity` should be visible for staff review but should not sound
the urgent alarm unless configured later by policy.

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

## Training Manifest Export

The training manifest is exported as JSONL, with one line per short labelled
video. Each line includes:

- `video_id`
- `filename`
- `label`
- `category`
- `scenario_name`
- `duration_seconds`
- `environment`
- `camera_angle`

For local training from the raw dataset, `dataset/training_manifest.jsonl` also
includes `relative_path`, which points to the checked-out Git LFS video file
under `dataset/raw`.

## Recording Preparation

Before recording Version 1 videos, use
`docs/simulated_video_recording_plan.md` and `dataset/metadata_template.csv` to
plan filenames, labels, and metadata rows. The first batch should be a small
balanced pilot batch so the upload, registration, and manifest export flow can
be tested before collecting more clips.

After recording, run `python scripts/generate_dataset_metadata.py` to scan
`dataset/raw`, calculate durations, and generate:

- `dataset/metadata.csv`
- `dataset/training_manifest.jsonl`

## Deferred High-Risk Simulations

The first baseline training dataset includes only behaviours that have recorded
clips in `dataset/raw`. The deferred high-risk labels should be added to model
training only after safe simulations or approved footage exist:

- `ligature_risk`
- `bleeding_visible`
- `sharp_object_detected`
- `attack_on_person`
- `head_banging`
- `property_damage`

These are already part of the alarm contract, but they should remain rule-only
or placeholder labels until clips are collected safely and reviewed.

Use `docs/high_risk_simulation_guide.md` before recording these clips.

This file is the first training handoff format. Later training scripts should
read the manifest instead of guessing labels from filenames.

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

## Observation Notes

The backend can generate draft observation notes from labelled behaviours or
future predictions. These notes must remain human-in-the-loop: they are drafts
for staff review, not final clinical documentation.
