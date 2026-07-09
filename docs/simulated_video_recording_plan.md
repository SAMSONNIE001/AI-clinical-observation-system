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
| fighting | 0 |
| attack_on_person | 0 |
| head_banging | 0 |
| property_damage | 0 |
| choking_simulation | 3 |
| ligature_risk | 0 |
| cutting_risk | 0 |
| bleeding_visible | 0 |
| blood_visible | 0 |
| sharp_object_detected | 0 |
| prolonged_inactivity | 3 |

This gives 33 pilot clips, plus no ligature-risk clips until the safe simulation
requirements are reviewed. Cutting, bleeding, blood, and sharp-object clips
should also wait until the safe simulation requirements are reviewed. Fighting
and attack-on-person clips should also wait until a safe no-contact simulation
plan is reviewed. Head-banging and property-damage clips should also wait until
a safe no-impact simulation plan is reviewed. Keep clips short, usually 5-20
seconds.

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
- Do not use real blades, knives, razors, broken glass, or any object that could
  injure someone. Use safe props only.
- Do not create real cuts or bleeding. Use fake blood or non-injury visual props
  only after review.
- Do not perform real falls. Use a staged low-risk movement onto a mat, bed, or
  padded surface if a fall class is recorded.
- Stop recording if a scenario feels unsafe or unclear.

## After Recording Each Clip

1. Rename the file using the filename pattern.
2. Add or update the row in `dataset/metadata_template.csv`.
3. Upload the video through the API when the backend is running.
4. Register the dataset video record with the same `video_id`, filename, label,
   category, scenario name, duration, environment, and camera angle.
5. Export the training manifest after the batch is registered.
