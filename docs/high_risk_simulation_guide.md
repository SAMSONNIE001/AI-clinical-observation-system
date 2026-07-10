# High-Risk Simulation Guide

Use this guide only for safe, controlled simulations. The goal is to help the
model learn visual warning patterns without creating real harm.

## Required Rules

- No real injury.
- No real weapons or dangerous sharp objects.
- No real ligature attempt.
- No real cutting.
- No real head impact.
- No real attack, restraint, fighting, or physical contact.
- No real property damage.
- No private patient data, staff data, documents, screens, or medication labels.
- Stop immediately if a scene feels unsafe.

## Target Labels

| Label | Folder | Safe scene idea |
| --- | --- | --- |
| `ligature_risk` | `dataset/raw/ligature_risk` | Non-harmful placeholder scene showing concerning setup cues only; no real neck contact or tightening |
| `bleeding_visible` | `dataset/raw/bleeding_visible` | Fake blood visible on cloth, tissue, sleeve, or skin-safe makeup |
| `sharp_object_detected` | `dataset/raw/sharp_object_detected` | Clearly visible safe prop representing scissors/blade/knife; no dangerous object |
| `attack_on_person` | `dataset/raw/attack_on_person` | No-contact staged fighting, lunge, or threatening posture; no contact and no real threat |
| `head_banging` | `dataset/raw/head_banging` | Mimed motion near a wall or cushion with no impact |
| `property_damage` | `dataset/raw/property_damage` | Mimed throwing/hitting or harmless soft prop; no real damage |

## Filename Pattern

Use exactly:

```text
<label>_<three_digit_number>.mp4
```

Examples:

```text
dataset/raw/attack_on_person/attack_on_person_001.mp4
dataset/raw/attack_on_person/attack_on_person_002.mp4
dataset/raw/head_banging/head_banging_001.mp4
dataset/raw/sharp_object_detected/sharp_object_detected_001.mp4
```

## First Target

Aim for 10 clips per high-risk label:

```text
9 labels x 10 clips = 90 clips
```

Use varied camera angles, lighting, distance, clothing, and backgrounds, but keep
each clip focused on one primary label.

## After Recording

Do not train immediately. First run validation:

- filename check
- folder check
- duration check
- video readability check
- metadata regeneration
- manifest regeneration
