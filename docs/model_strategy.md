# Model Strategy

The project should use pretrained vision models first, then fine-tune selected
parts with the simulated dataset. The simulated videos are not intended to train
a full behavior model from scratch.

## Pretrained Components

1. YOLO object detection
   - Detects people and object cues in each frame.
   - Current dangerous object mapping includes `knife` and `scissors`.
   - Later fine-tuning can add local examples of safe simulated sharp-object
     scenes.

2. MediaPipe pose movement analysis
   - Extracts body landmarks and movement summaries.
   - Useful for pacing, falling, prolonged inactivity, head movement, posture
     change, and aggressive movement cues.

3. Video behavior classifier
   - Best target: a pretrained video action model such as VideoMAE, TimeSformer,
     X3D, SlowFast, or another action-recognition backbone.
   - A plain CNN is weaker for this task because many labels depend on movement
     over time, not a single frame.
   - A CNN can still be used for frame-level visual cues, but the final behavior
     classifier should understand temporal motion.

## Current Code Path

The current pipeline combines:

- baseline video classifier,
- YOLO object detector when available,
- MediaPipe pose movement analyzer when available,
- alarm logic for high-risk behavior and dangerous objects.

This is the transition layer between the early baseline and the stronger
pretrained/fine-tuned system.

## Fine-Tuning Role Of Simulated Videos

The simulated videos should be used to:

- adapt pretrained models to the room, camera angle, lighting, and labels,
- validate false alarms and missed events,
- fine-tune a video classifier after the dataset is balanced,
- tune alarm thresholds separately from classifier accuracy.

## Recommended Next Classifier

Use a pretrained video action-recognition model rather than training a CNN from
scratch. The first fine-tuning target should be:

1. train/validation/test split from `dataset/training_manifest.jsonl`,
2. frame sampling from each video,
3. pretrained video backbone,
4. final classification head for the project labels,
5. separate alarm threshold calibration for high-risk classes.
