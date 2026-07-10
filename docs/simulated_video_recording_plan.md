# Simulated Video Recording Plan

This plan is for Version 1 only: short, safe, self-simulated clips with one
primary behaviour label per video.

## Record-Ready Gate

Start recording only when these are true:

- The behaviour list is final for Version 1.
- Filenames follow the agreed naming pattern.
- Each planned clip has a row in the metadata template.
- The recording environment is safe, controlled, and free of private data.
- Any fall or choking simulation is staged safely and does not involve real harm.
- Ligature risk is not recorded as a real attempt. Use only a safe, reviewed
  placeholder or staged non-harmful simulation after governance review.
- Cutting, bleeding, blood, and sharp-object scenes must not involve real injury
  or unsafe sharp objects.
- Fighting or attack scenes must not involve real violence, real contact, or
  unsafe restraint.
- Head-banging and property-damage scenes must not involve real impact, real
  injury, or real damage.
- Vomiting scenes must not involve real vomiting, unsafe ingestion, or
  biohazard exposure.

## Filename Pattern

Use:

```text
<behaviour_type>_<three_digit_number>.mp4
```

Examples:

- `sleeping_001.mp4`
- `walking_003.mp4`
- `pacing_002.mp4`
- `fall_001.mp4`

Use the same `video_id` without the extension:

```text
video_id = pacing_002
filename = pacing_002.mp4
```

## Folder Layout

Save each video inside the folder for its behaviour:

```text
dataset/raw/sleeping/sleeping_001.mp4
dataset/raw/sitting/sitting_001.mp4
dataset/raw/standing/standing_001.mp4
dataset/raw/walking/walking_001.mp4
dataset/raw/eating/eating_001.mp4
dataset/raw/reading/reading_001.mp4
dataset/raw/pacing/pacing_001.mp4
dataset/raw/fall/fall_001.mp4
dataset/raw/aggressive_movement/aggressive_movement_001.mp4
dataset/raw/choking_simulation/choking_simulation_001.mp4
dataset/raw/vomiting/vomiting_001.mp4
dataset/raw/prolonged_inactivity/prolonged_inactivity_001.mp4
```

When the high-risk simulation plan is reviewed, use the same structure:

```text
dataset/raw/ligature_risk/ligature_risk_001.mp4
dataset/raw/cutting_risk/cutting_risk_001.mp4
dataset/raw/bleeding_visible/bleeding_visible_001.mp4
dataset/raw/sharp_object_detected/sharp_object_detected_001.mp4
dataset/raw/attack_on_person/attack_on_person_001.mp4
dataset/raw/head_banging/head_banging_001.mp4
dataset/raw/property_damage/property_damage_001.mp4
```

Keep the filename exactly the same as the `filename` value in the metadata
template. The folder is only for organising the raw files.

## First Batch

Record a small balanced pilot batch before recording a larger dataset.

| Behaviour | Suggested clips |
| --- | ---: |
| sleeping | 3 |
| sitting | 3 |
| standing | 3 |
| walking | 3 |
| eating | 3 |
| reading | 3 |
| pacing | 3 |
| fall | 3 |
| aggressive_movement | 3 |
| choking_simulation | 3 |
| vomiting | 3 |
| prolonged_inactivity | 3 |

## High-Risk Fine-Tuning Batch

Record these only after the safety boundaries are clear. Start with 10 short
clips per label if possible.

| Behaviour | Suggested clips |
| --- | ---: |
| ligature_risk | 0 |
| cutting_risk | 0 |
| bleeding_visible | 0 |
| sharp_object_detected | 0 |
| attack_on_person | 0 |
| head_banging | 0 |
| property_damage | 0 |

Keep high-risk clips short, usually 5-20 seconds. Do not add these labels to
training until the files are validated and the metadata is regenerated.

## Scene Guidance

- Use one main behaviour per clip from start to finish.
- Keep the camera fixed where possible.
- Avoid filming faces when they are not needed.
- Remove names, documents, screens, medication labels, and private items from
  the scene.
- Avoid real clinical environments unless governance approval exists.
- Prefer consistent lighting and a stable camera angle for the first batch.

## Safety Boundaries

- Do not simulate real ligature attempts, self-harm, restraint, or real violence
  in Version 1.
- Do not stage real fights, hits, pushes, threats, or staff attacks. Use
  no-contact acting only after review.
- Do not hit your head against a wall, floor, object, or body part. Use
  no-impact acting only after review.
- Do not break or damage property. Use harmless props or mimed action only after
  review.
- Do not create choking simulations involving real airway obstruction.
- Do not create real vomiting. Use safe acting, props, or approved footage only
  after review.
- Do not use real blades, knives, razors, broken glass, or any object that could
  injure someone. Use safe props only.
- Do not create real cuts or bleeding. Use fake blood or non-injury visual props
  only after review.
- Do not perform real falls. Use a staged low-risk movement onto a mat, bed, or
  padded surface if a fall class is recorded.
- Stop recording if a scenario feels unsafe or unclear.

## After Recording Each Clip

1. Rename the file using the filename pattern.
2. Save it in the matching folder under `dataset/raw`.
3. Add or update the row in `dataset/metadata_template.csv`.
4. Upload the video through the API when the backend is running.
5. Register the dataset video record with the same `video_id`, filename, label,
   category, scenario name, duration, environment, and camera angle.
6. Export the training manifest after the batch is registered.
