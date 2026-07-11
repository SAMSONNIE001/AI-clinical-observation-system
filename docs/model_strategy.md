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
- structured risk engine for interpretable alarm decisions,
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
scratch. The first implemented fine-tuning target is TorchVision `r3d_18`,
pretrained on action-recognition video data, with a new classification head for
the project labels.

The detailed 18-label classifier is too hard for the current dataset size. Train
the grouped classifier first:

```text
python -m ml.training.train_video_action_classifier --label-mode grouped --epochs 8 --batch-size 1 --unfreeze-backbone --learning-rate 0.00005 --patience 2
```

The first run may download pretrained weights. The output checkpoint is written
to `ml/models/video_action_grouped_classifier.pt`, which the combined clinical
pipeline will use automatically when present.

The trainer saves the best validation epoch, not just the last epoch. This
matters when accuracy rises and then falls during later epochs.

For small datasets, the trainer uses class-balanced loss, light training
augmentation, random temporal sampling, weight decay, and early stopping. These
settings reduce overfitting but do not replace the need for hybrid YOLO,
MediaPipe, and rule-based safety checks.

## Structured Risk Engine

The current architecture now includes a structured risk engine. It takes signals
from YOLO, MediaPipe, and the grouped action classifier, then returns:

- risk group,
- risk level,
- alarm decision,
- reason list,
- observation summary.

This lets the live-camera system make safety decisions from interpretable
signals instead of depending only on a raw video classifier.

The fine-tuning workflow is:

1. train/validation/test split from `dataset/training_manifest.jsonl`,
2. frame sampling from each video,
3. grouped risk labels for the first reliable model,
4. pretrained `r3d_18` video backbone,
5. final classification head for grouped project labels,
6. separate alarm threshold calibration for high-risk groups.

The checkpoint file is ignored by Git by default because model weights can be
large. If a trained checkpoint needs to be versioned, use Git LFS deliberately.
